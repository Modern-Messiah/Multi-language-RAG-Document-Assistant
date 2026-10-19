# Reranking

Read this when you enable or debug a reranker, change a reranker model or alias, wire the
X-Reranker-* headers, or add a reranker provider. Who may send those headers and whose key they spend
is rag-llm-providers territory; this file covers what the reranker does inside retrieval. D labels
refer to the known-issues list in SKILL.md.

Contents:
1. Where it runs and what it sees
2. Providers, models and aliases
3. Server settings and per-request headers
4. Failure behaviour and how to tell it worked
5. Recipe: add a reranker provider

## 1. Where it runs and what it sees

`RAGChain._prepare`, after retrieval and the tenant-wide fallback:
`active_reranker = reranker if reranker is not None else self.reranker`, then, when chunks exist,
`active_reranker.rerank(search_query, docs, top_n=self.top_k)` inside a `rerank` span (input
`query`, `candidates_count`, `top_n`; output `count`, `sources`, `scores`). Without an active reranker
the chain cuts `docs[:top_k]` instead, which only happens for a RAGChain built without one (tests).
- The query is the condensed follow-up (`search_query`), not the user's literal text.
- The candidates are the whole fused list. With fetch_k = max(TOP_K_RESULTS, RETRIEVAL_CANDIDATES):
  up to max(4 x fetch_k, 25), 80 by default, with hybrid on; up to fetch_k, 20 by default, with hybrid off. Size
  latency and any per-request provider cost for that number, not for TOP_K_RESULTS.
- `top_n` is TOP_K_RESULTS. Real rerankers return new Document copies with `metadata["rerank_score"]`
  (a float); `_collect_sources` passes that score through for the first chunk of each file.
- The app always has a reranker object: `get_reranker(settings)` in the lifespan returns `NoOpReranker`
  when RERANKER_ENABLED=false, and `BaseReranker` defines no `__bool__`. So `fetch_k` is always
  max(TOP_K_RESULTS, RETRIEVAL_CANDIDATES) and a `rerank` span is recorded even with reranking off.

## 2. Providers, models and aliases

| Provider | Class | What it does | Default model |
| --- | --- | --- | --- |
| `none` (also disabled, or any unknown value) | `NoOpReranker` | `documents[:top_n]`, no score | - |
| `flashrank` | `FlashRankReranker` | local ONNX cross-encoder via `flashrank.Ranker`, imported lazily | `ms-marco-TinyBERT-L-2-v2` |
| `cohere` | `CohereReranker` | POST to `https://api.cohere.com/v2/rerank` with a Bearer key, `return_documents` false | `rerank-v3.5` |

`normalize_reranker_model(provider, name)` maps aliases (case-insensitive, trimmed); anything else
passes through unchanged, and empty means the provider default:
- flashrank: `multilingual`, `multi`, `ms-marco-multi-bert-l-12`, `multibert`, `bge`, `bge-m3` ->
  `ms-marco-MultiBERT-L-12`; `default`, `tiny`, `tinybert` -> `ms-marco-TinyBERT-L-2-v2`. The `bge`
  aliases give MultiBERT, not a BGE model.
- cohere: `multilingual`, `default`, `v3.5` -> `rerank-v3.5`.

FlashRank specifics: `flashrank` is not in `requirements.txt`, either lock file or the Docker image, so
in a stock install it never reranks (D15, KNOWN_ISSUES.md, section "Models, keys and reranking").
`app/rag/reranker.py` defines a stand-in `RerankRequest` when the package is missing, so the module
(which the lifespan imports) still loads and a ranker injected in a test still works. Initialised
rankers are cached per model name in the module-level `_RANKER_CACHE`; an import or init failure is
remembered only on that instance, and per-request instances are new each time, so they retry (and
warn) on every request.

Cohere specifics: the httpx timeout is a fixed 10 s, the URL cannot be set from Settings, and
`top_n` sent is `min(top_n, len(documents))`.

## 3. Server settings and per-request headers

Server-wide, read once by `get_reranker(settings)` in the lifespan (`app.state.reranker`):
RERANKER_ENABLED (`False`), RERANKER_PROVIDER (`none`; a free string, lower-cased - a typo such as
`flash-rank` silently means NoOp), RERANKER_MODEL (`""`, aliases apply), RERANKER_API_KEY (`""`).

Per request, `app/main.py` `_caller_reranker` calls `app/byok.py` `wanted_reranker(headers, settings)`:
- `X-Reranker-Provider` absent -> None, and the server reranker is used.
- `none`, `flashrank` or `cohere` (trimmed, case-insensitive) -> a fresh reranker from
  `get_reranker(settings, provider_override=..., api_key_override=..., model_override=...)`. The
  override skips the RERANKER_ENABLED check, and `none` replaces the server reranker with NoOp for
  that request.
- Any other provider -> 400 `Unknown reranker provider '<value>'. Supported: none, flashrank, cohere.`
- `cohere` with neither `X-Reranker-Key` nor RERANKER_API_KEY -> 400 `Cohere reranker requires an API
  key in X-Reranker-Key header.`; with no header key but a server key, the server key is used.
- `X-Reranker-Model` goes through the same aliases.

The policy consequences (works with reranking disabled, not limited by ALLOWED_MODEL_PROVIDERS, spends
the operator's Cohere key) are D14, owned by rag-llm-providers. The Streamlit "Cross-Encoder Reranker"
expander sends these headers from `asking_headers`; `evaluation/langfuse_eval.py` and
`evaluation/eval_sberquad_failures.py` take `--reranker` and `--reranker-model` (rag-evaluation).

## 4. Failure behaviour and how to tell it worked

A reranker never raises: every failure logs a WARNING and returns `documents[:top_n]` in the incoming
(dense or fused) order, because an exception in `_prepare` would turn the whole question into a 503.
- FlashRank: `Failed to initialize FlashRank (...); falling back to NoOp` (once per instance) or
  `FlashRank reranking failed: ...; using vector order`.
- Cohere: `CohereReranker has no API key; falling back to vector order`, or
  `Cohere rerank failed (...); falling back to vector order` for HTTP errors, timeouts and a rejected
  key; empty results also fall back. A refused caller key is therefore never reported to the caller.
- Success logs INFO `FlashRank reranked N candidates -> M top results (best_score=...)` or
  `Cohere reranked N candidates -> M results (best_score=...)`.

It worked if the sources carry a non-null `rerank_score` (/query always has the key, null when absent;
the stream's sources event omits the key when absent) and the `rerank` span's `scores` is non-empty. No score with a
provider selected means a fallback; read the WARNING logged under that request id.

## 5. Recipe: add a reranker provider

1. `app/rag/reranker.py`: subclass `BaseReranker`. `rerank(query, documents, top_n)` returns `[]` for
   no documents or `top_n <= 0`, at most `top_n` new Documents with copied metadata plus a float
   `rerank_score`, and on any failure a WARNING plus `documents[:top_n]`. `CohereReranker` is the
   template.
2. Import its SDK inside methods or behind try/except ImportError: the module is imported by the
   lifespan and by `wanted_reranker`.
3. Add a `DEFAULT_<X>_MODEL` constant and an alias branch in `normalize_reranker_model` if needed.
4. `get_reranker`: add a branch; read settings with `getattr(settings, name, default)` because
   `settings` may be None.
5. `app/byok.py` `wanted_reranker`: add the name to the allowed tuple and to the 400 text, and decide its
   key rule with the rag-llm-providers skill.
6. New settings: `app/config.py`, `.env.template` and the DOCUMENTATION.md Configuration table with the
   exact default (rag-config-and-docs); the lifespan already passes the whole Settings to `get_reranker`.
7. Clients and tools: the provider list in the Streamlit reranker expander (`frontend/streamlit_app.py`)
   and the `--reranker` choices of `evaluation/langfuse_eval.py` and `evaluation/eval_sberquad_failures.py`.
8. A hard dependency goes into `requirements.txt` with both locks regenerated (rag-development-guide);
   otherwise keep it optional like flashrank and say so in DOCUMENTATION.md.
9. Tests in `tests/test_reranker.py`: mock the transport (`patch("httpx.Client.post", ...)`) or the model
   object, and assert `rerank_score` is set - the offline socket guard makes an unmocked call fail into
   the fallback, which would otherwise pass silently. Add factory, header (400) and fallback tests, and a
   chain test like `test_rag_chain_ask_invokes_reranker_and_attaches_score`.
10. Run `python -m pytest tests/test_reranker.py tests/test_byok.py tests/test_streaming.py tests/test_docs_consistency.py -q`.
