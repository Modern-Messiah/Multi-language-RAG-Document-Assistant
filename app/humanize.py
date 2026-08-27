"""Numbers as a person reads them.

A module of its own, and a light one, because both the API and the two clients
need it: the API to word a limit, the clients to show usage. Until now it lived
in app.main, which the Streamlit process cannot import without pulling FastAPI,
openai and chromadb into a web page.
"""

_UNITS = (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024))


def human_size(num_bytes: int) -> str:
    """Render a byte count in the largest unit that keeps it readable.

    Formatting MB with "{:.0f}" alone reported a 1 KB limit as "0 MB"; a GB
    unit exists because a 1 GiB quota reads as "1 GB", not "1024 MB".
    """
    for unit, size in _UNITS:
        if num_bytes >= size:
            # One decimal, but "5.0 MB" reads worse than "5 MB".
            value = f"{num_bytes / size:.1f}".rstrip("0").rstrip(".")
            return f"{value} {unit}"
    return f"{num_bytes} bytes"


def describe_quota(quota) -> str:
    """One line of usage against the limits, for either client.

    "3 of 200 documents, 12 KB of 1 GB". A limit of 0 means that limit is off,
    and then only the usage is shown: "3 documents, 12 KB" - never "3 of 0".
    Returns "" for anything that is not a quota block, so a client talking to
    an older backend draws nothing rather than crashing.
    """
    if not isinstance(quota, dict):
        return ""

    documents = int(quota.get("documents") or 0)
    max_documents = int(quota.get("max_documents") or 0)
    used = int(quota.get("bytes") or 0)
    max_bytes = int(quota.get("max_bytes") or 0)

    if max_documents:
        documents_part = f"{documents} of {max_documents} documents"
    else:
        documents_part = f"{documents} document{'s' if documents != 1 else ''}"

    if max_bytes:
        bytes_part = f"{human_size(used)} of {human_size(max_bytes)}"
    else:
        bytes_part = human_size(used)

    return f"{documents_part}, {bytes_part}"
