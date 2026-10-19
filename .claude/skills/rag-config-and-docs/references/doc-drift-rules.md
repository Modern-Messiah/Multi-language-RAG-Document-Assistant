# Doc-drift rules: exact checks and traps

Every mechanical rule that ties the documents to the code, with the regex or
condition the test applies and the ways an innocent edit breaks it. Rule labels
(R1-R17, L, K, O, X) are this skill's shorthand; the test function names are the
source of truth. Unless noted, the rule lives in
`tests/test_docs_consistency.py`, which reads DOCUMENTATION.md, README.md and
.env.template as UTF-8 text.

Contents
1. Reading a failure
2. Configuration table and .env.template (R1-R4, R17)
3. Claims that were false once (R5-R7)
4. The chunking line (R8)
5. README and the bot (R9-R12)
6. The Compose probe (R13)
7. Commands and lint (R14, R15)
8. API reference headings (R16)
9. Languages (L), lock files (K), offline guarantee (O), other cross-file checks (X)
10. Agent docs (`tests/test_agent_docs.py`)
11. Not checked: keep in sync by hand

## 1. Reading a failure

| Message fragment | Rule | Usual cause |
| --- | --- | --- |
| settings missing from the Configuration table | R1 | no row, or the row has a trailing space or an empty last cell |
| documented variables that do not exist | R2 | a removed setting's row, or another table whose first cell is a backticked UPPER_CASE token |
| docs say ..., code default is ... / no backticked default in | R3 | `45` for a float, `true` for a bool, the default not the first backticked token, a pipe in the description |
| .env.template is missing / documents unknown variables | R4 | commented-out line, leading space, `export`, or a removed setting |
| never mentions BACKEND_API_KEY | R7 | the key's name dropped while rewriting a section |
| no line describes the chunking defaults / an assertion on the joined lines | R8 | a new line that mentions chunking and characters |
| README advertises /query (or /upload) | R9 | an API path or URL in README.md |
| is not in README.md / is missing from /help / could not find the /help text | R10 | new bot command, or the help block reformatted |
| assertion on the registered commands vs BOT_COMMANDS | R9 | a command added or removed without editing BOT_COMMANDS, or registered with single quotes |
| still names the old bot path / does not show how to run the bot | R11 | the entry point dropped from a doc or from the Compose file |
| does not show how to run the production configuration | R12 | the `-f docker-compose.yml` command removed |
| no healthcheck probe found / is not in the API reference / is probed without credentials | R13 | probe rewritten in another form, heading renamed, auth note moved down |
| the docs tell the reader to run `python -m ...` | R14 | typo or a module that was moved |
| not linted in CI | R15 | new top-level package |
| ... is missing from the API reference | R16 | new route without its heading |
| ERROR collecting tests/test_docs_consistency.py | R17 | a Settings field without a default |
| names no heading of / does not resolve to a file or directory | agent docs (section 10) | a link to a renamed or unwritten heading or file, in README.md or DOCUMENTATION.md too |
| citations that start with none of the KNOWN_ISSUES.md sections | agent docs (section 10) | a section cited by a name it does not have |
| the CLAUDE.md skills table needs one row | agent docs (section 10) | a new skill without its row, or a second row for one |

## 2. Configuration table and .env.template

### R1 - every setting has a row

Test: `test_every_setting_appears_in_the_config_table`.

```python
re.findall(r"^\| `([A-Z0-9_]+)` \|[^|]*\| (.+?) \|$", DOCUMENTATION, re.M)  # -> {NAME: last cell}
```

Write each row exactly as `` | `NAME` | Description. | `default` | ``: one space on
each side of every pipe, the name in backticks, and the line ending in ` |`.
Traps (each confirmed against the regex):
- A trailing space after the final pipe hides the row, and the test then reports
  the setting as missing rather than the row as malformed.
- A pipe inside the description, even escaped as `\|`, moves the cell boundary:
  the "default" cell then starts inside the description, and R3 compares the first
  backticked token after that pipe. It passes only by luck, so keep pipes out.
- Two spaces in the `` | ` `` prefix, no space before the final pipe, or an empty
  last cell also hide the row.
- Only `[A-Z0-9_]` names count; a lowercase or hyphenated first cell is not a row.
- The regex scans the whole file, not only `## Configuration`, and `dict()` keeps
  the last row for a duplicated name.

### R2 - no invented rows

Test: `test_config_table_has_no_invented_variables`. Every parsed row must be a
Settings field or in `CLIENT_ONLY_VARS = {"TELEGRAM_BOT_TOKEN", "BACKEND_URL"}`.
Trap: any table with three or more columns whose first cell is a backticked
UPPER_CASE token is parsed as a configuration row, wherever it sits in
DOCUMENTATION.md (with four columns the "default" becomes "col3 | col4"). Two-column
tables, and first cells such as `X-API-Key`, `.pdf` or `openai`, are not parsed.
Present per-feature knobs as prose or as a two-column table.

### R3 - documented defaults equal the code

Test: `test_documented_default_matches_the_code[NAME]`, one case per Settings
field except OPENAI_API_KEY (it has no default; its cell shows `-`).

```python
quoted = re.findall(r"`([^`]*)`", cell)                 # cell = the row's last cell
expected = "" if quoted[0] == '""' else quoted[0]
_normalise(expected) == _normalise(str(default))
# _normalise(v) = v.replace("\\", "/").lstrip("./").rstrip("/")
```

| Code default | Write | Fails |
| --- | --- | --- |
| `True` | `` `True` `` | `` `true` `` |
| `45.0` (float field) | `` `45.0` `` | `` `45` `` |
| `0.0` | `` `0.0` `` | `` `0` `` |
| `31457280` | `` `31457280` (30 MB) `` | `` `30 MB` `` |
| `""` | `` `""` `` | `-` or an empty cell |
| `Path("./data/chroma_db")` | `` `./data/chroma_db` `` or `` `data/chroma_db` `` | - |

- Only the first backticked token of the last cell is compared, so the default comes
  first; units and notes go after it, outside backticks.
- `lstrip("./")` strips any run of leading dots and slashes, so path spellings with
  or without `./` compare equal.
- Client-only rows (TELEGRAM_BOT_TOKEN, BACKEND_URL) are never compared: the
  clients read them (`clients/telegram_bot.py`, `clients/backend.py`), so their
  defaults drift silently (BACKEND_URL's code default is `DEFAULT_BACKEND_URL` in
  `clients/backend.py`).

### R4 - .env.template names every setting

Tests: `test_env_template_covers_every_setting`, `test_env_template_has_no_unknown_variables`.

```python
re.findall(r"^([A-Z0-9_]+)=", ENV_TEMPLATE, re.M)
```

- Every Settings field, OPENAI_API_KEY included, needs an uncommented `NAME=value`
  line starting at column 0. `# NAME=`, an indented line and `export NAME=` do not count.
- Any other `NAME=` line must be in CLIENT_ONLY_VARS.
- Values and comments are not checked. Booleans may be lowercase there (`true`);
  pydantic parses them.

### R17 - every field needs a default

At import the module pops every Settings variable from the environment and builds
`Settings(_env_file=None, openai_api_key="placeholder")` to parametrize R3. A field
without a default raises ValidationError at collection, and pytest stops the whole
run with a collection error. `tests/test_config.py` and many other tests also build
`Settings(_env_file=None, openai_api_key="k")`.

## 3. Claims that were false once

- R5 `test_docs_do_not_claim_user_id_is_optional`: DOCUMENTATION.md contains
  neither `user_id (optional)` nor the same phrase with a closing backtick after
  `user_id` (as in a code span followed by "(optional)").
- R6 `test_docs_do_not_mention_the_retired_streamlit_user_tenant`: `streamlit_user`,
  the web UI's old shared tenant, appears in neither README.md nor DOCUMENTATION.md.
- R7 `test_docs_describe_the_api_key_header`: `BACKEND_API_KEY` in both files,
  `X-API-Key` in DOCUMENTATION.md.

## 4. The chunking line (R8)

Test: `test_docs_state_the_real_chunking_defaults`.

```python
lines = [l for l in DOCUMENTATION.splitlines() if "hunking" in l and "haracter" in l]
described = " ".join(lines)
assert lines                                     # at least one such line
assert str(chunk_size) in described and str(chunk_overlap) in described   # "1000", "200"
assert "500" not in described and "50 char" not in described
```

- Case-sensitive substrings: "Chunking", "chunking", "Characters" all match;
  "CHUNKING" does not.
- Every such physical line in the whole file is joined. A new sentence such as
  "table-aware chunking keeps tables up to CHUNK_SIZE characters" must then carry
  1000 and 200 itself or the joined text must, and it must not contain "500"
  (also inside 1500, 2500 or 5000) or "50 char" (also inside "250 characters").
- If a line is not about the defaults, reflow the paragraph so "chunking" and
  "character" fall on different physical lines; Markdown renders it the same.
- Changing CHUNK_SIZE or CHUNK_OVERLAP means rewriting the defaults line too.

## 5. README and the bot

### R9 - no /query or /upload token in README, and the command set

Test: `test_docs_do_not_advertise_nonexistent_bot_commands`.

```python
re.findall(r"`?/(\w+)`?", README)             # any match equal to "upload" or "query" fails
set(re.findall(r'CommandHandler\("(\w+)"', bot_source)) == BOT_COMMANDS
BOT_COMMANDS = {"start", "help", "clear", "documents", "model"}
```

- Fails: `/query/stream`, `POST /upload`, `http://127.0.0.1:8000/query`,
  `/upload-file`, even "query/upload" in prose. Passes: `/query_stream`,
  `data/uploads/...`, `/documents/{file_hash}`.
- Name endpoints in README by what they do ("the streaming endpoint") and link to
  the API Reference in DOCUMENTATION.md, which has no such rule.
- Register commands as `CommandHandler("name", handler)` with double quotes in
  `clients/telegram_bot.py`; a single-quoted name is invisible and the set
  comparison fails. Edit BOT_COMMANDS whenever the set changes, and never call a
  command upload or query.

### R10 - every command is documented

Test: `test_every_bot_command_is_documented`.

```python
f"/{command}" in README                                       # every registered command
re.search(r"help_text = \(\n(.*?)\n    \)", bot_source, re.DOTALL)   # the /help block
f"/{command}" in help_block                                   # every command except help
```

Keep `help_text = (` followed directly by a newline, and the closing `)` alone on
its line indented exactly four spaces. A first string on the same line as the paren
gives "could not find the /help text"; a closing paren at another indent lets the
match run on to the next four-space `)` further down the file, so the check reads
the wrong text. DOCUMENTATION.md's Telegram usage list is not checked.

### R11, R12 - entry point and production command

- R11 `test_docs_reference_the_current_bot_entry_point`: `clients.telegram_bot` in
  README.md, DOCUMENTATION.md and `docker-compose.yml`, and the bot's path from before
  it moved into clients/, `telegram/bot.py`, in none of them.
- R12 `test_production_compose_command_is_documented`: `-f docker-compose.yml` in
  README.md and DOCUMENTATION.md, because a plain `docker compose up` also merges
  `docker-compose.override.yml` (bind mounts, `--reload`).

## 6. The Compose probe (R13)

Test: `test_the_compose_probe_is_the_endpoint_the_docs_describe`.

```python
probed = set(re.findall(r"urlopen\('http://localhost:8000(/[a-z]+)'", compose))
assert probed                                                  # "no healthcheck probe found"
f"### `GET {path}`" in DOCUMENTATION
"**No authentication**" in DOCUMENTATION.split(f"### `GET {path}`")[1][:400]
```

- Keep the healthcheck in the python urllib form with single quotes, `localhost`,
  port 8000 and a one-segment lowercase path. Double quotes, 127.0.0.1, a second
  segment or a trailing slash make the probe invisible. The slim image has no curl.
- `**No authentication**` must appear within the 400 characters after the first
  occurrence of the heading, so keep it in the first lines of the section.

## 7. Commands and lint

### R14 - documented `python -m` modules exist

Test: `test_every_documented_command_is_a_real_module` (README.md and DOCUMENTATION.md).

```python
found = set(re.findall(r"python -m ([\w.]+)", DOCUMENTATION + README))
ours = {d.name for d in ROOT.iterdir() if d.is_dir() and (d / "__init__.py").exists()}
# for each found module whose first segment is in ours:
#   ROOT/<a>/<b>.py or ROOT/<a>/<b>/__main__.py must exist; at least one must be documented
```

- Only the literal "python -m " spelling is matched in these two files; `python3 -m`
  and `.venv/Scripts/python.exe -m` are not seen here (the agent-docs test sees them,
  but only in agent docs).
- A bare package is not a runnable module: there is no `__main__.py` in app/, so
  "python -m" followed by just app fails.
- `frontend/` has no `__init__.py`; it runs as `streamlit run frontend/streamlit_app.py`.
- Flags are never checked. Compare every flag you write with `python -m <pkg>.<mod> --help`,
  except for `scripts.download_benchmarks`, which has no argument parser and starts
  downloading even when given `--help`.

### R15 - CI lints every package

Test: `test_the_linter_covers_every_package_of_ours`.

```python
linted = set(re.search(r"ruff check ([\w /]+)", ci_yml).group(1).split())
assert not ({packages with __init__.py} | {"frontend"}) - linted
```

Only the first `ruff check` line in `.github/workflows/ci.yml` is read. The copies
in the docs are compared by `tests/test_agent_docs.py` (section 10).

## 8. API reference headings (R16)

Test: `test_every_route_has_a_section_in_the_api_reference`.

```python
re.findall(r'@(?:router|application)\.(get|post|delete|put|patch)\("([^"]+)"', main_py)
# each (method, path) needs "### `METHOD /path`" in DOCUMENTATION.md,
# e.g. ### `DELETE /documents/{file_hash}`
```

Only decorators on objects named `router` or `application` in `app/main.py` with a
double-quoted literal path are seen. `add_api_route`, single quotes, a route in
another module or a differently named router escape the check, and the reference
then misses the route without a failure. The section body (fields, headers,
status codes) is not checked.

## 9. Languages, locks, offline, cross-file

### L - one language table (`tests/test_language_consistency.py`)

- `SUPPORTED_LANGUAGES[0] == AUTO_LANGUAGE` ("Auto") and the rest equal the keys of
  `LANG_RULES` in `app/rag/languages.py`; Auto is not a rule; no duplicates; no
  empty rule; `rule_for` returns the rule and falls back to `AUTO_RULE` for unknown names.
- `chain.LANG_RULES is languages.LANG_RULES`; `clients.backend.SUPPORTED_LANGUAGES`
  is the same object; the bot's `LANGUAGES` equals the list and its keyboard has at
  most three buttons per row.
- `frontend/streamlit_app.py` mentions `SUPPORTED_LANGUAGES` and assigns `LANG_FLAGS`
  as a literal dict (read with `ast.literal_eval`) whose keys equal the supported
  languages, Auto included. Adding the flag is required, not optional.
- The language lists in README.md and DOCUMENTATION.md prose are not checked.

### K - lock files (`tests/test_lockfile.py`)

Header flags of both locks, `==` pins with sha256 hashes, extras kept, declared
specifiers satisfied, dev lock a superset at equal versions, posthog below 4, no
pytest or ruff in the runtime lock. It does not catch a removed requirement, a
hand-edited transitive pin or a reordered flag; only the CI lock job's diff does.
The regeneration procedure belongs to the rag-development-guide skill.

### O - offline guarantee (`tests/test_offline_guarantee.py`)

Outbound `socket.create_connection` and `sock.connect` raise RuntimeError matching
"must stay" (keep that wording in `tests/conftest.py`), loopback works, fake
embeddings are deterministic, the `api` fixture's chat client is
`tests.conftest.FakeChatClient`, `OPENAI_API_KEY` is `test-key-not-real`, and
`RAGChain.ask` without a user_id raises ValueError("user_id is required..."). See
the rag-testing skill.

### X - other cross-file checks

- `tests/test_docx_markdown.py::test_nothing_hardcodes_a_format_list_any_more`: the
  string SUPPORTED_EXTENSIONS appears in `app/main.py`, `frontend/streamlit_app.py`
  and `clients/telegram_bot.py`.
- `tests/test_byok.py::test_the_offered_providers_are_the_ones_documented`: the key
  set of `PROVIDERS` in `app/byok.py` is exactly openai, anthropic, gemini, deepseek,
  kimi, kimi-cn. `test_each_provider_gets_its_own_endpoint` lists one (provider, host)
  case per non-openai provider by hand; a new provider is untested until you add one.
- Value pins, which fail when a default changes: `tests/test_quotas.py::test_quotas_ship_on`
  (200 documents, 1 GiB), `tests/test_mmr.py::test_mmr_ships_disabled` (1.0),
  `tests/test_relevance.py::test_the_threshold_defaults_to_disabled` and
  `tests/test_retrieval_quality.py::test_the_shipped_threshold_is_still_disabled` (0.0),
  `tests/test_feedback.py::test_feedback_ships_enabled` and, for the client with
  FEEDBACK_ENABLED unset, `test_the_buttons_are_offered_by_default` in the same file.
- Mirror checks, which fail when a copy falls out of step:
  `tests/test_clients_backend.py::test_max_file_size_default_matches_the_backend`
  (`DEFAULT_MAX_FILE_SIZE` in `clients/backend.py`) and
  `tests/test_backup.py::test_the_fallbacks_are_settings_own_defaults` (the paths
  `app/backup.py` `storage_locations` uses without a Settings instance).
- No test pins the defaults of HYBRID_SEARCH_ENABLED, METADATA_FILTERING_ENABLED,
  TABLE_AWARE_CHUNKING, RERANKER_ENABLED or LANGFUSE_ENABLED beyond R3.

## 10. Agent docs (`tests/test_agent_docs.py`)

Agent docs are CLAUDE.md, KNOWN_ISSUES.md and every `.md` file under
`.claude/skills/` (references included). The link, section-citation and lint checks
also read README.md and DOCUMENTATION.md. The exact patterns are module constants in
the test (`PYTHON_M`, `RUFF_CHECK`, `LINK`, `LINK_DEFINITION`, `SECTION_CITATION`,
`NOT_A_PATH`, `REPO_DIRS`, `ROOT_FILES`, `ROOT_FAMILIES`, `MACHINE_PATH`); read them
there. Inline markup is read per block (a paragraph, list item, heading or table
row), so a code span or link wrapped onto the next line still counts; fenced code is
skipped by the link and citation checks but read by the path and command checks.

- `test_every_skill_directory_has_a_skill_md`: every directory directly under
  `.claude/skills/` has a SKILL.md, spelled exactly.
- `test_skill_frontmatter[<skill>]`: the file starts with `---` (no byte order
  mark), has a closing `---`, and holds one-line `key: value` entries (no folded or
  multi-line values). An unquoted value may not contain ": " or " #", end with ":",
  or start with a YAML indicator character; wrap such a value in double quotes.
  `name` equals the directory name, matches lowercase-digits-hyphens and is at most
  64 characters; `description` is non-empty, at most 1024 characters, without angle
  brackets.
- `test_skill_files_are_at_most_500_lines`: SKILL.md only, frontmatter included;
  reference files have no size check.
- `test_claude_md_points_to_every_skill`: CLAUDE.md has exactly one table row per
  skill directory that starts with a link to its SKILL.md, shaped
  `| [<name>](.claude/skills/<name>/SKILL.md) | when to use it |`, with the skill
  name as the link text; the path named elsewhere in prose does not count.
  `test_every_skill_path_in_claude_md_exists`: every `.claude/skills/<name>/SKILL.md`
  path anywhere in CLAUDE.md exists.
- `test_relative_links_resolve[<doc>]`, for README.md, DOCUMENTATION.md and every
  agent doc: each inline link or image outside code, and each reference-style link
  definition, resolves from the linking file's directory, with exact case. http(s)
  and mailto targets are skipped; empty targets, backslashes and absolute paths fail.
  A `#anchor`, alone or after a path to a `.md` file, must name a heading of that
  file the way GitHub slugs it (lowercased, punctuation and emoji dropped, each space
  a hyphen, so `## Installation & Setup` is `#installation--setup`, and a repeated
  heading gets `-1`, `-2`) or an `<a id>` or `<a name>` target. A link to a section
  that was renamed or is not written yet fails.
- `test_backticked_repo_paths_exist[<doc>]`: every word of an inline code span or a
  fenced block (split at whitespace, quotes, brackets, commas and `=`, so the path in
  a command is read alone) that names a path under app, clients, frontend, tests,
  evaluation, scripts, .github or .claude (it must contain a slash), one of the root
  files (README.md, DOCUMENTATION.md, CLAUDE.md, KNOWN_ISSUES.md, Dockerfile, the
  three Compose files, .env.template, pyproject.toml, the requirements and lock
  files, .dockerignore, .gitignore, LICENSE) or an invented sibling of one (another
  Compose, requirements, Dockerfile or `.env.` name) must exist, exact case,
  directories included. Words with any of `< > * { } $` are skipped as placeholders,
  globs or variables; a `:line`, `::test` or `#anchor` suffix, a leading `./`,
  surrounding parentheses or quotes and trailing punctuation are stripped. Other
  directories (data/, .venv/) are never checked, and neither are README.md and
  DOCUMENTATION.md.
- `test_known_issues_sections_cited_by_name_exist`: in README.md, DOCUMENTATION.md
  and every agent doc, the capitalised words after "KNOWN_ISSUES.md, section" (or
  "sections"), quoted, linked or plain, must start with one of the `##` headings of
  KNOWN_ISSUES.md. A placeholder such as "section <name>" is skipped. Renaming a
  section means updating every citation in the same change.
- `test_every_python_m_command_is_a_real_module`: every `-m` run through python,
  python3, python3.10, python.exe or the Windows launcher py (interpreter options such
  as `-X utf8` may come before `-m`), in prose and code alike, whose module starts
  with a package of ours, or is one of our modules written without its leading
  package (the sweep script named without `scripts.`), must resolve to a file, and
  at least one such command must exist. A placeholder or glob right after the module
  (as in `scripts.<name>`) is skipped; other tools (pytest, ruff) are not checked.
- `test_every_full_lint_command_matches_ci`: every `ruff check` inside a code span or
  fenced block of README.md, DOCUMENTATION.md or an agent doc whose targets are all
  bare names (options and their values dropped; `app/` and `./app` count as `app`)
  lists exactly CI's set. A file or path target (`ruff check tests/test_agent_docs.py`)
  or `ruff check .` is not compared, and neither is "ruff check" in running prose.
- `test_human_docs_show_the_full_lint_command[README.md]` and `[DOCUMENTATION.md]`:
  each shows the full command at least once.
- `test_no_machine_specific_paths[<doc>]`: no macOS or Linux home-directory prefix,
  no WSL mount of a Windows drive, no Windows user-profile path and no Claude Code
  per-session temp directory anywhere in an agent doc, fenced code included. The
  pattern would match a quoted example, so describe such paths in words;
  `~/.venvs/<name>` and `.venv/Scripts/python.exe` pass.

Traps for anyone writing agent docs: an invented example path in code fails, in a
command too (write `tests/test_<area>.py`); a wrong-module `python -m` fails even
when written as a counter-example; a partial lint command with bare package names
fails (lint a single file by its path instead); a link to a sibling skill that does
not exist yet, or to a heading not written yet, fails, so name siblings in prose.

## 11. Not checked: keep in sync by hand

- Configuration-table descriptions and every .env.template value and comment.
- Client-only defaults (`clients/backend.py`, `clients/telegram_bot.py`) against
  their table rows.
- Numbers in prose other than the chunking line: quotas, timeouts, sizes, model
  names in examples, retrieval candidate counts.
- API Reference bodies: request fields, headers, response fields, status codes.
- Route mentions in README.md other than the /query and /upload ban.
- Flags of documented commands, and `python -m` spellings other than the ones above.
- Bot command descriptions, DOCUMENTATION.md's Telegram usage list, the format,
  language and provider lists in README.md, DOCUMENTATION.md and the .env.template
  comment.
- Docker prose (probe, volumes, `.dockerignore`), the Directory Structure tree,
  the Continuous Integration section text, the architecture diagram and components.
- Whether the README.md skills table lists every skill (its links are checked, its
  completeness is not), and anchors inside external URLs.
- Symbols named in backticks (only paths are checked), and whether any agent-doc
  or KNOWN_ISSUES.md statement is still true.
- Test counts and timings: do not write them.
