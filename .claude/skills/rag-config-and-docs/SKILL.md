---
name: rag-config-and-docs
description: Adds or changes a setting end to end (app/config.py Settings, .env.template, the DOCUMENTATION.md Configuration table, lifespan wiring) and keeps README.md, DOCUMENTATION.md, KNOWN_ISSUES.md, CLAUDE.md and the .claude/skills files in sync with the code under the mechanical doc-drift tests. Use when touching an environment variable or default, editing any of those documents, or when tests/test_docs_consistency.py or tests/test_agent_docs.py fails.
---

# Settings and documentation

All configuration is one pydantic-settings class, `Settings` in `app/config.py`. README.md
serves users, DOCUMENTATION.md developers and operators, KNOWN_ISSUES.md lists verified
defects, and CLAUDE.md plus `.claude/skills/` guide coding agents. The parts that drifted
before (false defaults, a retired tenant, undocumented features) are pinned by
`tests/test_docs_consistency.py`, `tests/test_language_consistency.py` and
`tests/test_agent_docs.py`; everything else stays true only if the change that alters the code
updates it too. This skill is that checklist.

## Use this skill for / not for

For: adding, renaming or removing a setting or a default; which documents a change must touch;
editing README.md, DOCUMENTATION.md, CLAUDE.md, KNOWN_ISSUES.md or a skill; a red doc test.

Not for the content, which the area skill owns: retrieval knobs (rag-retrieval-pipeline),
chunking and formats (rag-document-ingestion), providers and the LLM endpoint
(rag-llm-providers), routes and API Reference bodies (rag-api-endpoints), bot commands and UI
controls (rag-clients), locks, CI and commits (rag-development-guide), writing tests
(rag-testing), measuring a default (rag-evaluation), backups, sweep, volumes (rag-operations).

## Map

- `app/config.py` - `Settings` (env var = upper-cased field name), two `model_validator`s
  (`_api_key_is_header_safe`, `_chunk_overlap_fits_in_chunk`), `get_settings` (`lru_cache`).
- `app/main.py` - `create_app(settings)` stores it on `app.state.settings`; `lifespan` builds
  every component from it; handlers read `request.app.state.settings`.
- `.env.template` - one `NAME=value` line per field plus client-only TELEGRAM_BOT_TOKEN, BACKEND_URL.
- `clients/backend.py` - what the clients read themselves: BACKEND_URL (`backend_url`),
  BACKEND_API_KEY (`api_headers`), MAX_FILE_SIZE (`max_file_bytes`, `DEFAULT_MAX_FILE_SIZE`),
  FEEDBACK_ENABLED (`feedback_enabled`). The bot reads TELEGRAM_BOT_TOKEN in `clients/telegram_bot.py`.
- `tests/test_docs_consistency.py` (R1-R17, `CLIENT_ONLY_VARS`, `BOT_COMMANDS`),
  `tests/test_language_consistency.py` (L), `tests/test_agent_docs.py`, `tests/test_config.py`
  (validators, wiring), `tests/conftest.py` (`make_settings`; `_isolated_environment` strips
  every Settings variable from the environment once per session).

## How a setting reaches the code

1. `Settings` takes, highest priority first, constructor arguments, the process environment,
   then `.env` in the current working directory (start the backend from the repo root).
   `extra="ignore"`: a misspelled key is silently ignored and the default applies.
2. `.env` is never exported to `os.environ`, so `os.getenv` in app code misses values that
   exist only there. That once crashed startup (`test_dotenv_only_startup_works` guards it), and
   it is why every knob is a field and components receive values explicitly.
3. `uvicorn app.main:app` builds the app from the cached `get_settings()`: restart after editing `.env`.
4. `lifespan` passes fields into constructors (`TextChunker`, `EmbeddingsManager`, `RAGChain`)
   or hands the whole object to a factory (`store_from_settings`, `get_tracer`, `get_reranker`).
5. The clients never import Settings: they call `load_dotenv()` (searching upward from their own
   directory) and read a few variables with `os.getenv`, mostly through `clients/backend.py`.
6. `python -m scripts.backup` and `python -m scripts.restore` read only the process environment
   with Settings' field defaults as fallbacks (`app/backup.py` `storage_locations`), never `.env`
   (KNOWN_ISSUES.md, section "Operations: backup, restore and sweep").

## Recipe: add a setting

1. `app/config.py`: a typed field with a default under the matching `# --- ... ---` comment,
   constrained with `Field(default=..., ge=..., le=..., min_length=...)`. Above it, a comment
   giving the reason for the default: what a wrong value does and how loudly (policy below). A
   field without a default turns the whole test run into a collection error (R17).
2. Cross-field rules go in a `@model_validator(mode="after")` that raises ValueError naming the
   env vars, like `_chunk_overlap_fits_in_chunk`: failing at startup beats failing on first use.
3. `.env.template`: an uncommented `NAME=value` line at column 0 under the matching `# ===`
   header, with a comment above it. Use the code default unless the template should differ on
   purpose (the two secrets do); nothing checks values.
4. DOCUMENTATION.md `## Configuration`: a row `` | `NAME` | Description. | `default` | `` next
   to related rows, the default written as `str()` of the code value (`True`, `45.0`, `""`,
   `31457280` (30 MB)). A trailing space hides the row (R1 then reports it missing); a pipe in
   the description, even escaped, shifts the default cell (R3).
5. Wire it: pass the field from `lifespan` into the component's constructor, or read it from
   `request.app.state.settings` in a handler. Add no new `os.getenv` fallback; the ones in
   `RAGChain.__init__` serve only direct construction.
6. A path field: give it a `tmp_path` value in the `make_settings` defaults so no test writes
   into the repository; a new data directory also needs the per-change checklist below.
7. Needed by the clients too: mirror it in `clients/backend.py` with `os.getenv` and the same
   default, pinned by a test like `test_max_file_size_default_matches_the_backend`. A variable
   only the clients read is not a Settings field: add it to `CLIENT_ONLY_VARS` in
   `tests/test_docs_consistency.py`, then give it a row and a template line.
8. Tests (rag-testing skill): bounds and validators in `tests/test_config.py`
   (`pytest.raises(ValidationError)` around `Settings(_env_file=None, openai_api_key="k", ...)`),
   a wiring assertion like `test_settings_are_wired_into_components`, and a ships-on or
   ships-off test when the default is a product decision.
9. Prose: the feature's DOCUMENTATION.md section, README.md Key Features if users notice it,
   CLAUDE.md or a skill if they name the setting.

## Recipe: change, rename or remove a setting

1. Change the field default and rewrite the comment that justified the old one.
2. Change the `.env.template` value too. The template sets every knob, so a deployment made
   with `cp .env.template .env` keeps the template value whatever the code says, and existing
   `.env` files keep theirs: say so in the commit body.
3. Change the table cell, then hunt the old value in prose:
   `grep -rn "<old value>" README.md DOCUMENTATION.md CLAUDE.md KNOWN_ISSUES.md .claude/skills .env.template`.
   Besides the cell (R3) only the chunking line (R8) is checked; quotas, timeouts, sizes and
   model names in examples are not.
4. Update the tests that pin the value (below) and the client mirrors `DEFAULT_MAX_FILE_SIZE`
   and the `return True` fallback in `feedback_enabled`. Component constructors repeat some
   defaults for direct construction (`TextChunker`, `RAGChain.__init__`, `EmbeddingsManager`,
   the `DEFAULT_*` constants in `app/rag/hybrid.py`); the app always passes the Settings value,
   but tests that build components directly use them, so grep and decide.
5. Special cases: the DeepSeek branch in `lifespan` compares MODEL_NAME with the literal
   `gpt-4o-mini`; a new EMBEDDING_MODEL makes startup refuse an existing collection, and it or
   a new COLLECTION_NAME makes restore refuse older archives without `--force`; CHUNK_SIZE and
   CHUNK_OVERLAP affect new uploads only and appear in the R8 line; path defaults must keep
   matching the Compose mounts under `/app/data` and the Dockerfile `mkdir`.
6. Rename or removal: field, template line and row go together (a stale row fails R2, a stale
   line R4); grep the old NAME across the docs, `clients/`, `scripts/`, `evaluation/` and the
   Compose files. A deployment still setting the old name silently gets the default, so say so
   in the commit body.

## Knob-default policy

Written in the comments of `app/config.py`:
- A knob whose wrong value fails silently ships off until someone measures it.
  RELEVANCE_THRESHOLD is `0.0` because a guessed cut-off drops relevant context unnoticed;
  MMR_LAMBDA is `1.0` because the measured recall gain came only after precision had fallen.
- A knob whose overshoot is loud may ship on with a generous value: MAX_DOCUMENTS_PER_USER (200)
  and MAX_BYTES_PER_USER (1 GiB) refuse with a 413 carrying the numbers. FEEDBACK_ENABLED is on
  because a record is written only when a user presses a button. Optional integrations ship
  off: LANGFUSE_ENABLED, RERANKER_ENABLED.
- Exceptions, not precedent: HYBRID_SEARCH_ENABLED, METADATA_FILTERING_ENABLED and
  TABLE_AWARE_CHUNKING ship `True` although their failures are silent, and their comments
  record no measurement; KNOWN_ISSUES.md sections "Retrieval and search" and "Document
  ingestion" list what that costs.
- Changing a silent-failure default needs a before/after measurement in the commit body
  (rag-evaluation skill).

Tests that pin a default value (change them only with a deliberate default change):
`tests/test_quotas.py::test_quotas_ship_on`, `tests/test_mmr.py::test_mmr_ships_disabled`,
`tests/test_relevance.py::test_the_threshold_defaults_to_disabled`,
`tests/test_retrieval_quality.py::test_the_shipped_threshold_is_still_disabled`,
`tests/test_feedback.py::test_feedback_ships_enabled` and its client twin
`test_the_buttons_are_offered_by_default`. Tests that keep a copy in step: R3, R8,
`tests/test_clients_backend.py::test_max_file_size_default_matches_the_backend` and
`tests/test_backup.py::test_the_fallbacks_are_settings_own_defaults`.

## The doc-drift contract

Exact regexes, failure messages and traps: [references/doc-drift-rules.md](references/doc-drift-rules.md);
read it when a doc test fails or before restructuring a document. Rules live in
`tests/test_docs_consistency.py` unless noted:

- R1 every Settings field has a DOCUMENTATION.md row `` | `NAME` | Description. | `default` | ``:
  single spaces around pipes, no pipe in the description, no trailing space.
- R2 every parsed row is a field or in `CLIENT_ONLY_VARS`, so no other table of three or more
  columns anywhere in the file may start with a backticked UPPER_CASE cell.
- R3 the first backticked token of the last cell equals `str(default)` (`True`, `45.0`, `""`);
  paths compare without a leading `./`. OPENAI_API_KEY (no default) is skipped.
- R4 `.env.template` has an uncommented `NAME=` at column 0 for every field, OPENAI_API_KEY
  included, and no other names but the client-only two.
- R5-R7 DOCUMENTATION.md never contains `user_id (optional)`; neither doc contains
  `streamlit_user`; both mention BACKEND_API_KEY; DOCUMENTATION.md mentions X-API-Key.
- R8 the DOCUMENTATION.md lines containing "hunking" and "haracter" state 1000 and 200 and
  contain neither "500" nor "50 char" (1500 and "250 characters" fail too).
- R9 README.md has no `/query` or `/upload` token anywhere, URLs and `/query/stream`
  included; the registered `CommandHandler("...")` names equal `BOT_COMMANDS`.
- R10 every command appears as `/name` in README.md and, except help, in the `help_text`
  block of `clients/telegram_bot.py`.
- R11 `clients.telegram_bot` in README.md, DOCUMENTATION.md and `docker-compose.yml`, and
  `telegram/bot.py` in none; R12 both docs contain `-f docker-compose.yml`.
- R13 the Compose healthcheck keeps the `urlopen('http://localhost:8000/<path>'` form, that
  path has a `` ### `GET /path` `` heading, and `**No authentication**` follows within 400 characters.
- R14 every `python -m <pkg>.<mod>` of ours in README.md and DOCUMENTATION.md is a real
  module; R15 the first `ruff check` line in `.github/workflows/ci.yml` covers every package
  with `__init__.py`, plus frontend.
- R16 every `@router` or `@application` route in `app/main.py` has a `` ### `METHOD /path` ``
  heading; R17 every field has a default.
- L (`tests/test_language_consistency.py`) one language table, `app/rag/languages.py`;
  `LANG_FLAGS` in `frontend/streamlit_app.py` is a literal dict naming exactly the languages.
- K `tests/test_lockfile.py` (lock headers, pins, hashes; rag-development-guide); O
  `tests/test_offline_guarantee.py` (no network in tests; rag-testing).
- X SUPPORTED_EXTENSIONS named in `app/main.py`, `frontend/streamlit_app.py` and
  `clients/telegram_bot.py`; the exact `PROVIDERS` set in `tests/test_byok.py`; the default pins above.

Not checked, so keep it true by hand: table descriptions and `.env.template` values and
comments; client-only defaults (compare with `clients/backend.py`); numbers in prose; API
Reference bodies (fields, headers, status codes); flags of documented commands; format,
language and provider lists in prose; DOCUMENTATION.md's Telegram usage list; Docker prose,
the Directory Structure tree, the CI section and the diagram; whether README.md's skills table
lists every skill; whether any statement in CLAUDE.md, a skill or KNOWN_ISSUES.md is still true.

## README.md constraints

- No `/query` or `/upload` token anywhere (R9): name endpoints in words and link to the API
  Reference in DOCUMENTATION.md, which has no such rule.
- Mentions every bot command (`/start`, `/help`, `/clear`, `/documents`, `/model`),
  `BACKEND_API_KEY`, `clients.telegram_bot` and `-f docker-compose.yml`; never
  `telegram/bot.py` or `streamlit_user`.
- Every `python -m` module exists (R14). It shows the CI lint command
  `ruff check app frontend clients evaluation scripts tests` at least once, and every other
  full lint command must list the same packages (agent-docs test).
- Known defects get one line and "see KNOWN_ISSUES.md, section <name>". Links and `#anchors`
  must resolve (agent-docs test); no test checks that the skills table lists every skill.

## Where things live

DOCUMENTATION.md (`grep -n '^##' DOCUMENTATION.md` prints the outline): `## Key Features`;
`## System Architecture` with `### Components`; `## Installation & Setup` with `### Development`
and `### Dependencies and the lock files`; `## Docker Deployment`; `## Usage` (`### 3. Telegram
Bot` lists bot commands); `## API Reference`, one `` ### `METHOD /path` `` per route; one `##`
per feature, such as `## Your own key and model`, `## Quotas and retention`, `## Backup and
restore`, `## Supported formats`, `## Supported languages`; `## Configuration` (the table) and
retrieval subsections such as `### Relevance filtering`; then `## Benchmarks and Langfuse
experiments`, `## Known issues` (a short summary linking KNOWN_ISSUES.md), `## Working with AI
coding agents` (the skill list), the Directory Structure and `## Continuous Integration`. A new
feature gets its own `##`; renaming one breaks links into it (README.md links many by anchor).

KNOWN_ISSUES.md has fixed sections, in this order, that other documents cite by name:
"Retrieval and search", "Document ingestion", "Models, keys and reranking", "Tracing and
observability", "Operations: backup, restore and sweep", "Web UI", "Telegram bot", "Evaluation
and benchmarks", "Development and tests". An entry is a `### <symptom>` heading with
**Severity** (high, medium or low), **Affects**, **What happens**, **Workaround** and **Fix
direction** bullets; unreproduced risks are `####` entries under `### Unverified risks` (Operations).

## Agent docs upkeep

- CLAUDE.md is loaded into every session: keep it near 150 lines, its commands exact and its
  invariants true. Its skills table needs exactly one row per skill directory, shaped
  `` | [rag-testing](.claude/skills/rag-testing/SKILL.md) | when to use it | `` with the skill
  name as link text, and every `.claude/skills/<name>/SKILL.md` path in it must exist (tested).
- A skill is `.claude/skills/<name>/SKILL.md` with two frontmatter keys: `name` (the directory
  name; lowercase letters, digits, hyphens) and `description` (one line, third person, what it
  does and when to use it with concrete trigger terms, at most 600 characters by house rule
  though the test allows 1024, no angle brackets, double-quoted if it contains ": " or " #" or
  starts with a special character).
- Body 120-300 lines, never over 450 (the test fails a SKILL.md above 500). Long material goes
  to `references/<topic>.md` one level deep, linked with a "read this when" hint; a reference
  over about 100 lines opens with a contents list. Usual order: purpose, use for / not for, map,
  how it works, recipes, invariants, pitfalls and known issues, verify, related skills.
- Adding, renaming or re-scoping a skill:
  1. `.claude/skills/<name>/SKILL.md` with the frontmatter above; the directory name is the name.
  2. Its "Use this skill for / not for" names the siblings it borders, and those siblings' lists
     and descriptions point back, so two descriptions never claim the same task.
  3. The CLAUDE.md skills-table row (tested), the README.md skills table and the list in
     DOCUMENTATION.md, section "Working with AI coding agents" (both by hand).
  4. `grep -rn "<old-name>" CLAUDE.md README.md DOCUMENTATION.md .claude/skills` for a rename.
  5. `pytest tests/test_agent_docs.py -q`.
- After a code change, touch agent docs only where the change makes a statement false or extends
  a list they enumerate (providers, bot commands, settings, the test index). Do not add mentions
  of the change elsewhere: every extra agent-doc edit is review load on a small pull request.
- Content: repo-relative path plus symbol, never line numbers; placeholders like
  `tests/test_<area>.py` in examples, because every repo path in code must exist; siblings
  by name, not by link; only commands you ran; no counts, timings, dates or machine-specific
  paths. `.claude/` is excluded from the Docker build context.

`tests/test_agent_docs.py` treats CLAUDE.md, KNOWN_ISSUES.md and every `.md` under
`.claude/skills/` as agent docs and checks: skill frontmatter and size, the CLAUDE.md skills
table, repo paths in code spans and fenced code, `python -m` modules in any spelling and
machine-specific paths. Three checks also read README.md and DOCUMENTATION.md: relative links
and their `#anchors` (exact case, GitHub heading slugs), KNOWN_ISSUES.md sections cited by
name (against its `##` headings), and every full `ruff check` against CI's. A counter-example
fails like a mistake: never quote a wrong module, path or partial lint command.

## Known issues: adding and fixing

- Add an entry only for a defect you reproduced, in the right section and format. Elsewhere
  write one line plus "see KNOWN_ISSUES.md, section <name>": no copy of the entry, no invented anchor.
- Fixing one, in the same change: a regression test that fails without the fix (revert the fix
  once to watch it fail), the entry deleted, and its pointers removed or rewritten as current
  behaviour (`grep -rn "<distinctive words>" README.md DOCUMENTATION.md CLAUDE.md .claude/skills`).
  Keep the nine section headings even when one empties; citations of them are tested.

## Per-change doc checklist

| Change (owner skill) | A test fails if you miss | Update by hand |
| --- | --- | --- |
| New route (rag-api-endpoints) | `` ### `METHOD /path` `` heading (R16); `**No authentication**` near it if Compose probes it (R13) | request fields, headers, status codes; Components; README describes it in words (R9) |
| New setting (this skill) | row and default (R1-R3), template line (R4), a default in code (R17) | description, the feature section, README if user-visible, client mirror |
| New bot command (rag-clients) | `CommandHandler("name", ...)` plus `BOT_COMMANDS` (R9); `/name` in README and in the help block (R10) | `### 3. Telegram Bot` in DOCUMENTATION.md; never name it upload or query |
| New file format (rag-document-ingestion) | SUPPORTED_EXTENSIONS still named in the three files (X); a new parser dependency in both locks (K) | Key Features in both docs, the README bot line, `## Supported formats` |
| New BYOK provider (rag-llm-providers) | the exact `PROVIDERS` set in `tests/test_byok.py` (X) | a case in `test_each_provider_gets_its_own_endpoint`; provider table in `## Your own key and model`; README; the ALLOWED_MODEL_PROVIDERS comment in `.env.template`; `provider_placeholders` in the web UI |
| New answer language (rag-retrieval-pipeline) | `LANG_FLAGS` entry (L) | Key Features in both docs, `## Supported languages` |
| New `python -m` module (rag-development-guide) | the documented module exists (R14, agent-docs test); a new top-level package joins the CI ruff line (R15) and then every full lint command in the docs | flags (run `--help`), its DOCUMENTATION.md section, Directory Structure, CLAUDE.md commands, `.dockerignore` (tests and evaluation stay out of the image, scripts ship) |
| New data directory (rag-operations) | only through its path setting (R1-R4) | `make_settings`; Dockerfile `mkdir -p` before the `chown` (`data/feedback` is missing there, an unverified risk in KNOWN_ISSUES.md); a named volume in `docker-compose.yml` (service and top level) and `docker-compose.override.yml`; `.gitignore`, `.dockerignore`; `app/backup.py` if it must be backed up; Docker, backup and Directory Structure sections |

## Style

- Plain hyphens, never em or en dashes; with GNU grep this prints nothing:
  `grep -rnP '\x{2013}|\x{2014}' README.md DOCUMENTATION.md CLAUDE.md KNOWN_ISSUES.md .claude/skills`.
- Measured claims: say what was measured and how; no unmeasured numbers or "fast", no test
  counts, timings, dates or "recently".
- Explain why in one clause, as the comments in `app/config.py` do: a rule without its reason
  is the one the next editor breaks. A known defect is current behaviour plus a workaround.

## Verify

From the repository root in a dev venv (per-OS and WSL setup: rag-development-guide). Move a
`.env` in the repo root or a parent folder aside first: the web UI and bot tests load it, which
can change results (KNOWN_ISSUES.md, section "Development and tests").

```bash
python -m pytest tests/test_docs_consistency.py tests/test_language_consistency.py tests/test_agent_docs.py -q
python -m pytest tests/test_config.py tests/test_quotas.py tests/test_mmr.py tests/test_relevance.py tests/test_retrieval_quality.py tests/test_feedback.py tests/test_clients_backend.py tests/test_backup.py -q
ruff check app frontend clients evaluation scripts tests
python -m pytest -q
```

One case: `python -m pytest "tests/test_docs_consistency.py::test_documented_default_matches_the_code[CHUNK_SIZE]" -q`.
For every command you document, compare its flags with `python -m <pkg>.<mod> --help`, except
`scripts.download_benchmarks`: it has no argument parser and starts downloading from the
network even when given `--help` (KNOWN_ISSUES.md, section "Evaluation and benchmarks").

## Related skills

rag-architecture (where to change what), rag-development-guide (environment, lint, locks, CI,
commits), rag-testing (the tests this skill asks for), rag-change-review (review checklist
including this contract); content owners: "Use this skill for / not for".
