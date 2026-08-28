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

**The endpoint is chosen from a list, never supplied.** A caller names a
provider - "anthropic", "gemini", "deepseek", "kimi" - and the URL comes from
the table below, which this file owns. Letting a caller pass the URL itself
would make this backend fetch anything it is told to, internal addresses
included. An operator who wants fewer providers than the list offers sets
ALLOWED_MODEL_PROVIDERS.

Every provider here speaks OpenAI's chat completions protocol, which is why one
client library reaches all of them. Their endpoints were checked by asking each
one with a deliberately invalid key; what came back is in the table.
"""
import logging
import re
from typing import Optional

import httpx
from openai import OpenAI

logger = logging.getLogger(__name__)

KEY_HEADER = "X-Model-Key"
MODEL_HEADER = "X-Model"
PROVIDER_HEADER = "X-Model-Provider"

# Where each provider's OpenAI-compatible chat completions live. The client
# library appends "chat/completions", so these are base URLs.
#
# "openai" is spelled None rather than a URL: it means the deployment's own
# OPENAI_BASE_URL, which an operator may already have pointed at a gateway
# (OpenRouter, LiteLLM, vLLM, Azure). Naming it explicitly is how a caller says
# "the usual one" after another provider has been set.
PROVIDERS = {
    "openai": None,
    "anthropic": "https://api.anthropic.com/v1/",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "deepseek": "https://api.deepseek.com/v1",
    "kimi": "https://api.moonshot.ai/v1",
    # Moonshot runs separate estates; a key from one is not valid at the other.
    "kimi-cn": "https://api.moonshot.cn/v1",
}

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


def allowed_providers(settings=None) -> list:
    """Provider names this deployment permits, in table order.

    ALLOWED_MODEL_PROVIDERS empty means all of them. An operator on a network
    with egress rules decides where their backend may connect; that is not a
    caller's choice to make.
    """
    configured = (getattr(settings, "allowed_model_providers", "") or "").strip()
    if not configured:
        return list(PROVIDERS)

    named = {name.strip().lower() for name in configured.split(",") if name.strip()}
    return [name for name in PROVIDERS if name in named]


def wanted(headers, settings=None) -> tuple:
    """The (key, model, provider) a caller asked for, validated. Any may be None.

    Raises BringYourOwnKeyError with wording safe to show the caller. The key
    itself is never echoed back, not even truncated: whatever they sent, the
    fix is the same.
    """
    key = (headers.get(KEY_HEADER) or "").strip()
    model = (headers.get(MODEL_HEADER) or "").strip()
    provider = (headers.get(PROVIDER_HEADER) or "").strip().lower()

    if key and not _KEY.match(key):
        raise BringYourOwnKeyError(
            "The API key you sent is not shaped like one. Check for a stray "
            "space or a truncated paste."
        )
    if model and not _MODEL.match(model):
        raise BringYourOwnKeyError(
            "That model name has characters a model name cannot have."
        )
    if (model or provider) and not key:
        raise BringYourOwnKeyError(
            "Choosing a model or a provider needs your own API key: send it "
            "too, or send neither and the assistant answers with its own."
        )
    if provider:
        permitted = allowed_providers(settings)
        if provider not in permitted:
            raise BringYourOwnKeyError(
                f"Unknown or disallowed provider. This deployment offers: "
                f"{', '.join(permitted)}."
            )

    return (key or None), (model or None), (provider or None)


def client_for(key: str, settings, provider: str = None) -> OpenAI:
    """An OpenAI-protocol client on the caller's key, at the named provider.

    The caller closes it. Timeouts and retries stay the deployment's, and the
    URL comes from PROVIDERS rather than from the request, so a caller's key
    cannot make the backend wait longer than the operator allows or talk to an
    address the operator never listed.
    """
    options = {
        "timeout": settings.openai_timeout,
        "max_retries": settings.openai_max_retries,
    }
    base_url = PROVIDERS.get(provider) if provider else None
    if base_url:
        options["base_url"] = base_url
    elif settings.openai_base_url:
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
    if status == 402:
        return "Your account with that provider is out of balance."
    if status is not None:
        # Anything else the provider answered with. There is one because
        # providers disagree about what a bad key is: OpenAI, Anthropic,
        # DeepSeek and Moonshot say 401, Gemini says 400. Without this, a
        # Gemini key typo fell through to "Query failed" and looked like the
        # operator's fault.
        return (
            f"The provider refused the request (HTTP {status}). Check the key "
            "and that the model name is one that provider offers."
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
