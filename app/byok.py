"""Answering with the caller's own API key and model of choice.

"Bring your own key": a user hands the assistant a key they pay for and names
the model they want answers from. The deployment's key still pays for indexing,
because the vector store is one collection bound to one embedding model - Chroma
fixes the dimension per collection, and embeddings.py refuses to open a
collection built by a different model, since vectors from two models are not
comparable. So the choice on offer is the *answer* model, which is the one a
person can actually tell apart.

Three decisions worth stating, because each closes something.

**The key is never stored.** It arrives in a header, is used for one request,
and the client that holds it is closed when that request ends. Nothing writes it
to disk, and no cache keyed on it keeps it alive in the process: a pool per key
would be faster, and would mean the secret outlives the request that carried it.
`user_id` is unauthenticated client input, so a stored key would also be a key
anyone holding the shared secret could spend.

**A model may only be chosen together with a key.** Otherwise the choice is a
way to spend the operator's money on a costlier model than they configured.

**The base URL is not negotiable.** Letting a caller name the endpoint would
make this backend fetch any URL it is told to, internal addresses included; the
deployment's OPENAI_BASE_URL stands.
"""
import logging
import re
from typing import Optional

import httpx
from openai import OpenAI

logger = logging.getLogger(__name__)

KEY_HEADER = "X-Model-Key"
MODEL_HEADER = "X-Model"

# Header values are latin-1 on the wire while clients send UTF-8, so a key with
# anything outside this shape could never round-trip anyway. Bounded because it
# is untrusted input that reaches an HTTP client.
#
# The lower bound is 8 rather than something closer to the ~51 characters an
# OpenAI key has: OPENAI_BASE_URL may point at a self-hosted gateway where the
# token is whatever its operator chose, and refusing a short but legitimate one
# would be this file inventing a rule the provider does not have. Eight still
# catches the common mistake, a paste that was cut off.
_KEY = re.compile(r"^[A-Za-z0-9._\-]{8,256}$")

# "gpt-4o-mini", "gpt-4.1", "o3-mini", and the "vendor/model" form gateways use.
_MODEL = re.compile(r"^[A-Za-z0-9._:/\-]{1,80}$")


class BringYourOwnKeyError(Exception):
    """Something the caller sent is unusable, and they can fix it."""


def wanted(headers) -> tuple:
    """The (key, model) a caller asked for, validated. Both may be None.

    Raises BringYourOwnKeyError with wording safe to show the caller. The key
    itself is never echoed back, not even truncated: whatever they sent, the
    fix is the same.
    """
    key = (headers.get(KEY_HEADER) or "").strip()
    model = (headers.get(MODEL_HEADER) or "").strip()

    if key and not _KEY.match(key):
        raise BringYourOwnKeyError(
            "The API key you sent is not shaped like one. Check for a stray "
            "space or a truncated paste."
        )
    if model and not _MODEL.match(model):
        raise BringYourOwnKeyError(
            "That model name has characters a model name cannot have."
        )
    if model and not key:
        raise BringYourOwnKeyError(
            "Choosing a model needs your own API key: send both, or neither "
            "and the assistant answers with its configured model."
        )

    return (key or None), (model or None)


def client_for(key: str, settings) -> OpenAI:
    """An OpenAI client on the caller's key, on this deployment's endpoint.

    The caller closes it. Every transport setting except the key is the
    deployment's, so a user's key cannot make the backend talk to somewhere
    else or wait longer than the operator allows.
    """
    options = {
        "timeout": settings.openai_timeout,
        "max_retries": settings.openai_max_retries,
    }
    if settings.openai_base_url:
        options["base_url"] = settings.openai_base_url

    return OpenAI(
        api_key=key,
        http_client=httpx.Client(trust_env=False),
        **options,
    )


def describe_upstream_refusal(exc) -> Optional[str]:
    """What to tell a caller whose own key was refused, or None.

    A 401 from the provider is the caller's problem here, not the operator's -
    the opposite of every other 401 in this system - so it must not reach the
    clients' "contact the operator" wording. The provider's own message is not
    passed through: it can carry account details, and it says nothing the
    caller cannot work out from "your key was rejected".
    """
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return "The provider rejected the API key you sent."
    if status == 404:
        return (
            "The provider does not offer that model to your key. Check the "
            "model name."
        )
    if status == 429:
        return (
            "Your API key is out of quota or rate limited by the provider. "
            "This is on your account, not on the assistant."
        )
    return None


def close_quietly(client) -> None:
    """Let go of the caller's key.

    Not optional: a client per request with no close leaks a connection pool
    every time, which is how the test suite ran out of file descriptors once
    already.
    """
    if client is None:
        return
    try:
        client.close()
    except Exception:
        logger.warning("Could not close the per-request model client")
