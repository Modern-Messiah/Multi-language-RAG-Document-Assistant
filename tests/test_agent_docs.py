"""The agent docs must not drift from the repository either.

CLAUDE.md, KNOWN_ISSUES.md and the skills in .claude/skills/ are instructions
that coding agents act on rather than question: an agent runs the command and
opens the path it is given. CLAUDE.md once sent agents to a tool installed at a
path on one developer's machine, and README.md and DOCUMENTATION.md showed a
lint command that skipped a package CI lints. These tests pin the mechanical
part of the contract - skill frontmatter and size, the skills table in
CLAUDE.md, links and their #anchors (README.md and DOCUMENTATION.md link into
the agent docs, so theirs count too), KNOWN_ISSUES.md sections cited by name,
repo paths in code, `python -m` modules, the lint command and machine-specific
paths - so the next drift fails CI instead of misdirecting an agent. Whether
the prose is true is still a reviewer's job.
"""
import functools
import os
import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / ".claude" / "skills"


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


@functools.lru_cache(maxsize=None)
def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@functools.lru_cache(maxsize=None)
def _entries(directory: Path) -> frozenset:
    try:
        return frozenset(os.listdir(directory))
    except OSError:  # missing, or a file
        return frozenset()


def _exists(path: Path) -> bool:
    """Whether `path` exists inside the repository, spelled exactly as on disk.

    Path.exists() ignores case on Windows and macOS, so a link to `Readme.md`
    would pass there and fail on the Linux CI runner. normpath, unlike
    resolve(), keeps the spelling it was given.
    """
    try:
        parts = Path(os.path.normpath(path)).relative_to(ROOT).parts
    except ValueError:  # outside the repository
        return False
    directory = ROOT
    for part in parts:
        if part not in _entries(directory):
            return False
        directory = directory / part
    return True


def _skill_dirs() -> list:
    """Every directory directly under .claude/skills/; each one is a skill."""
    if not SKILLS.is_dir():
        return []
    return sorted(item for item in SKILLS.iterdir() if item.is_dir())


def _agent_docs() -> list:
    """CLAUDE.md, KNOWN_ISSUES.md once it exists, and every .md file of the skills."""
    docs = [ROOT / name for name in ("CLAUDE.md", "KNOWN_ISSUES.md") if _exists(ROOT / name)]
    if SKILLS.is_dir():
        docs += sorted(SKILLS.rglob("*.md"))
    return docs


SKILL_DIRS = _skill_dirs()
AGENT_DOCS = _agent_docs()
# README.md and DOCUMENTATION.md link to the skills, KNOWN_ISSUES.md and each
# other's sections, so their links and citations are checked as well.
LINKING_DOCS = [ROOT / "README.md", ROOT / "DOCUMENTATION.md", *AGENT_DOCS]


# =========================
# Reading markdown
# =========================

FENCE = re.compile(r"\s*(`{3,}|~{3,})(.*)")
# A list item, heading, table row or quote starts a block of its own, so a code
# span or a link never continues across one. Headings and table rows are one
# line long.
BLOCK_START = re.compile(r"\s*(?:[-*+]\s|\d+[.)]\s|#{1,6}(?:\s|$)|\||>)")
ONE_LINE_BLOCK = re.compile(r"\s*(?:#{1,6}(?:\s|$)|\|)")
CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)", re.DOTALL)


def _lines(text: str):
    """(number, line, is_code) per line; fenced blocks, fences included, are code."""
    fence = None
    for number, line in enumerate(text.splitlines(), start=1):
        match = FENCE.match(line)
        if fence is None:
            # A backtick fence's info string cannot contain a backtick, so
            # ```code``` on one line is a code span, not an opening fence.
            if match and not (match.group(1)[0] == "`" and "`" in match.group(2)):
                fence = match.group(1)
                yield number, line, True
            else:
                yield number, line, False
        else:
            closes = (
                match is not None
                and match.group(1)[0] == fence[0]
                and len(match.group(1)) >= len(fence)
                and not match.group(2).strip()
            )
            if closes:
                fence = None
            yield number, line, True


def _blocks(text: str):
    """(first line number, lines joined by newlines) of each paragraph, list
    item, heading and table row outside fenced code.

    Markdown lets a code span or a link run on to the next line of the same
    paragraph, and prose here is wrapped, so inline markup is read per block.
    Markdown renders neither inside a fenced block.
    """
    block, start = [], 0
    for number, line, is_code in _lines(text):
        if block and (is_code or not line.strip() or BLOCK_START.match(line)):
            yield start, "\n".join(block)
            block = []
        if is_code or not line.strip():
            continue
        if not block:
            start = number
        block.append(line)
        if ONE_LINE_BLOCK.match(line):
            yield start, "\n".join(block)
            block = []
    if block:
        yield start, "\n".join(block)


def _code_spans(text: str):
    """(number, content) of every inline code span; a wrapped span is unwrapped."""
    for start, block in _blocks(text):
        for match in CODE_SPAN.finditer(block):
            number = start + block.count("\n", 0, match.start())
            yield number, match.group(2).replace("\n", " ").strip()


def _fenced_lines(text: str):
    """(number, line) of every line of fenced code.

    A line ending in a backslash is joined to the next one, so a command reads
    the way a shell runs it.
    """
    pending, first = "", None
    for number, line, is_code in _lines(text):
        if not is_code:
            continue
        if line.rstrip().endswith("\\"):
            pending += line.rstrip()[:-1] + " "
            first = first or number
            continue
        yield first or number, pending + line
        pending, first = "", None
    if pending:
        yield first, pending


def _code(text: str) -> list:
    """(number, code) of every inline code span and every line of fenced code."""
    return sorted([*_code_spans(text), *_fenced_lines(text)])


# =========================
# Skill files
# =========================

ENTRY = re.compile(r"([A-Za-z][\w-]*):(?:[ \t]+(.*?))?[ \t]*")
SKILL_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _scalar(raw: str) -> str:
    """The string YAML reads from a one-line value; ValueError if it would not.

    Quoted values are unescaped. An unquoted value is where YAML surprises:
    ": " inside it is a syntax error, " #" starts a comment that silently cuts
    the text, and a leading indicator character is syntax, not text.
    """
    if raw.startswith('"'):
        match = re.fullmatch(r'"((?:[^"\\]|\\.)*)"(?:[ \t]+#.*)?', raw)
        if not match:
            raise ValueError('the double-quoted value is not closed or holds an unescaped "')
        return re.sub(r"\\(.)", r"\1", match.group(1))
    if raw.startswith("'"):
        match = re.fullmatch(r"'((?:[^']|'')*)'(?:[ \t]+#.*)?", raw)
        if not match:
            raise ValueError("the single-quoted value is not closed or holds a lone ' (write '')")
        return match.group(1).replace("''", "'")
    if (raw[:1], raw[-1:]) in (("[", "]"), ("{", "}")):
        return raw  # a flow list or map, e.g. `allowed-tools: [Read, Grep]`
    if raw and (raw[0] in "[]{}>|*&!%@`#," or raw[:2] in ("- ", "? ", ": ")):
        raise ValueError(
            f"an unquoted value cannot start with {raw[0]!r}; wrap it in double quotes"
        )
    if ": " in raw or raw.endswith(":") or " #" in raw:
        raise ValueError(
            "an unquoted value cannot contain ': ' or ' #' or end with ':'; "
            "wrap it in double quotes"
        )
    return raw


def _frontmatter(text: str) -> dict:
    """{key: value} of the frontmatter block a skill file starts with.

    Read by hand rather than with PyYAML, which is only a transitive
    dependency: skills use one-line `key: value` entries, and anything else -
    a folded or multi-line value, a nested list - is rejected instead of half
    understood. Raises ValueError naming the line it cannot read.
    """
    lines = text.splitlines()
    if lines and lines[0].startswith("﻿"):
        raise ValueError("starts with a byte order mark; save it as UTF-8 without BOM")
    if not lines or lines[0].rstrip() != "---":
        raise ValueError(f"must start with a '---' line, not {(lines or [''])[0]!r}")
    closing = next((i for i, line in enumerate(lines) if i and line.rstrip() == "---"), None)
    if closing is None:
        raise ValueError("the frontmatter block has no closing '---' line")

    meta = {}
    for number, line in enumerate(lines[1:closing], start=2):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        entry = ENTRY.fullmatch(line)
        if not entry:
            raise ValueError(f"line {number} is not a one-line `key: value` entry: {line!r}")
        key = entry.group(1)
        if key in meta:
            raise ValueError(f"line {number}: `{key}` is set twice")
        try:
            meta[key] = _scalar(entry.group(2) or "")
        except ValueError as error:
            raise ValueError(f"line {number} ({key}): {error}") from None
    return meta


def test_every_skill_directory_has_a_skill_md():
    assert SKILL_DIRS, ".claude/skills/ holds no skill directories, so these tests prove nothing"

    missing = [f"{_rel(d)}/SKILL.md" for d in SKILL_DIRS if not _exists(d / "SKILL.md")]
    assert not missing, f"skill directories without a SKILL.md, which Claude Code skips: {missing}"


@pytest.mark.parametrize("skill", [d.name for d in SKILL_DIRS])
def test_skill_frontmatter(skill):
    """Claude Code picks a skill by its name and description alone, and a
    frontmatter block it cannot parse leaves the skill without either."""
    path = SKILLS / skill / "SKILL.md"
    label = _rel(path)
    assert _exists(path), f"{label} is missing"
    try:
        meta = _frontmatter(_read(path))
    except ValueError as error:
        meta, problem = None, f"{label}: {error}"
    if meta is None:  # failed outside the except block, so without a chained traceback
        pytest.fail(problem, pytrace=False)

    for key in ("name", "description"):
        assert key in meta, f"{label}: the frontmatter has no `{key}`"

    name = meta["name"]
    assert name == skill, f"{label}: name {name!r} differs from the directory name {skill!r}"
    assert SKILL_NAME.fullmatch(name), (
        f"{label}: name {name!r} must be lowercase letters and digits joined by single hyphens"
    )
    assert len(name) <= 64, f"{label}: the name is {len(name)} characters, the limit is 64"

    description = meta["description"]
    assert description.strip(), f"{label}: the description is empty"
    assert len(description) <= 1024, (
        f"{label}: the description is {len(description)} characters, the limit is 1024"
    )
    brackets = sorted(set(description) & {"<", ">"})
    assert not brackets, (
        f"{label}: the description contains {' and '.join(brackets)}, which the skill "
        "format does not allow"
    )


def test_skill_files_are_at_most_500_lines():
    """The whole SKILL.md enters the context once the skill triggers; reference
    material belongs in references/<topic>.md, which is read on demand."""
    too_long = {
        _rel(path): count
        for path in (d / "SKILL.md" for d in SKILL_DIRS)
        if _exists(path) and (count := len(_read(path).splitlines())) > 500
    }
    assert not too_long, f"SKILL.md files over 500 lines: {too_long}"


# =========================
# The skills table in CLAUDE.md
# =========================

SKILL_PATH = re.compile(r"\.claude/skills/([\w.-]+)/SKILL\.md")
# A row of the table: `| [<name>](.claude/skills/<name>/SKILL.md) | when to use it |`.
SKILL_ROW = re.compile(r"\|\s*\[([^\]]+)\]\(\.claude/skills/([\w.-]+)/SKILL\.md\)\s*\|")


def _skills_named_in_claude_md() -> set:
    claude_md = ROOT / "CLAUDE.md"
    assert _exists(claude_md), "CLAUDE.md is missing"
    return set(SKILL_PATH.findall(_read(claude_md)))


def _skills_table() -> list:
    """(number, link text, skill) for each CLAUDE.md table row that starts
    with a link to a SKILL.md; a path named anywhere else is not a row."""
    claude_md = ROOT / "CLAUDE.md"
    assert _exists(claude_md), "CLAUDE.md is missing"
    return [
        (number, row.group(1).strip("`* "), row.group(2))
        for number, line, is_code in _lines(_read(claude_md))
        if not is_code and (row := SKILL_ROW.match(line.strip()))
    ]


def test_claude_md_points_to_every_skill():
    """CLAUDE.md is the map an agent starts from; a skill missing from it is
    easy to overlook, and the table is updated by hand. Only a table row
    counts: the same path in prose elsewhere would hide a dropped row."""
    rows = _skills_table()
    named = [skill for _, _, skill in rows]

    problems = [f"no row for {d.name}" for d in SKILL_DIRS if d.name not in named]
    problems += [
        f"{named.count(skill)} rows for {skill}"
        for skill in sorted(set(named)) if named.count(skill) > 1
    ]
    problems += [
        f"line {number}: the link text {text!r} is not the skill it links, {skill!r}"
        for number, text, skill in rows if text != skill
    ]
    assert not problems, (
        "the CLAUDE.md skills table needs one row "
        "`| [<name>](.claude/skills/<name>/SKILL.md) | when to use it |` per skill:\n"
        + "\n".join(problems)
    )


def test_every_skill_path_in_claude_md_exists():
    broken = sorted(
        f".claude/skills/{name}/SKILL.md"
        for name in _skills_named_in_claude_md()
        if not _exists(SKILLS / name / "SKILL.md")
    )
    assert not broken, f"CLAUDE.md points to skills that do not exist: {broken}"


# =========================
# Links, anchors and paths
# =========================

LINK = re.compile(r"\]\(\s*(<[^<>\n]*>|[^)\s]*)")
# A reference-style link definition, `[label]: target`; `[^1]:` is a footnote.
LINK_DEFINITION = re.compile(r" {0,3}\[(?!\^)[^\]]+\]:[ \t]*(<[^<>\n]*>|\S+)")
EXTERNAL = re.compile(r"(?:https?|mailto):", re.IGNORECASE)
HEADING = re.compile(r" {0,3}(#{1,6})(?:[ \t]+(.*?))??(?:[ \t]+#+)?[ \t]*")
SETEXT_UNDERLINE = re.compile(r" {0,3}(?:=+|-+)[ \t]*")
HTML_ANCHOR = re.compile(r"<a\s[^>]*?\b(?:name|id)\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)


def _links(text: str):
    """(number, target) of every inline link, image and link definition outside code."""
    for start, block in _blocks(text):
        # Blank out code spans but keep the offsets: `[a](b)` in code is no link.
        prose = CODE_SPAN.sub(lambda span: re.sub(r"[^\n]", " ", span.group(0)), block)
        for match in LINK.finditer(prose):
            yield start + prose.count("\n", 0, match.start()), match.group(1)
        # Link definitions can only open a paragraph, one per line.
        for offset, line in enumerate(block.split("\n")):
            definition = LINK_DEFINITION.match(line)
            if not definition:
                break
            yield start + offset, definition.group(1)


def _slug(heading: str) -> str:
    """The anchor GitHub gives a heading: its text lowercased, punctuation and
    symbols (emoji included) dropped and each space turned into a hyphen, so
    "Installation & Setup" becomes installation--setup."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # a link keeps its text
    text = re.sub(r"<[^>]*>", "", text).replace("`", "").replace("*", "")
    kept = "".join(
        char for char in text.strip().lower()
        if char in " -_"
        or unicodedata.category(char)[0] in "LM"
        or unicodedata.category(char) in ("Nd", "Nl")
    )
    return kept.replace(" ", "-")


@functools.lru_cache(maxsize=None)
def _anchors(path: Path) -> frozenset:
    """Every #anchor GitHub renders for a markdown file, lowercased: one per
    heading (a repeated one gets -1, -2, ...) plus <a id> and <a name> targets."""
    text = _read(path)
    seen, previous = {}, None
    for _, line, is_code in _lines(text):
        heading = None
        if not is_code:
            atx = HEADING.fullmatch(line)
            if atx:
                heading = atx.group(2) or ""
            elif previous is not None and SETEXT_UNDERLINE.fullmatch(line):
                heading = previous
        prose = not is_code and line.strip() and not BLOCK_START.match(line)
        previous = line if prose and heading is None else None
        if heading is not None:
            slug = anchor = _slug(heading)
            while anchor in seen:  # github-slugger's numbering
                seen[slug] += 1
                anchor = f"{slug}-{seen[slug]}"
            seen[anchor] = 0
    return frozenset([*seen, *(name.lower() for name in HTML_ANCHOR.findall(text))])


def _broken_links(doc: Path, text: str) -> list:
    """(number, problem) for each relative link in `text` that resolves to
    nothing, or to a markdown file without the #anchor it names."""
    broken = []
    for number, raw in sorted(_links(text)):
        target = raw.strip("<>")
        if EXTERNAL.match(target):
            continue
        path, _, anchor = target.partition("#")
        path = unquote(path)
        destination = Path(os.path.normpath(doc.parent / path)) if path else doc
        if not target:
            problem = "is empty"
        elif "\\" in path:
            problem = "uses backslashes; write / in links"
        elif path.startswith("/"):
            problem = f"is absolute; link relative to {_rel(doc)}"
        elif not _exists(destination):
            problem = "does not resolve to a file or directory"
        elif (
            anchor
            and destination.suffix == ".md"
            and destination.is_file()
            and unquote(anchor).lower() not in _anchors(destination)
        ):
            problem = f"names no heading of {_rel(destination)}"
        else:
            continue
        broken.append((number, f"({raw}) {problem}"))
    return broken


# Only paths under these directories are checked: data/ and .venv/ are created
# at run time and do not exist in a checkout.
REPO_DIRS = {"app", "clients", "frontend", "tests", "evaluation", "scripts", ".github", ".claude"}
ROOT_FILES = {
    "README.md", "DOCUMENTATION.md", "CLAUDE.md", "KNOWN_ISSUES.md", "Dockerfile",
    "docker-compose.yml", "docker-compose.override.yml", "docker-compose.langfuse.yml",
    ".env.template", "pyproject.toml", "requirements.txt", "requirements-dev.txt",
    "requirements.lock", "requirements-dev.lock", ".dockerignore", ".gitignore", "LICENSE",
}
# Root files that are easy to invent a sibling of: `docker-compose.prod.yml`,
# `requirements-dev.in`, `Dockerfile.dev`, `.env.example`.
ROOT_FAMILIES = re.compile(
    r"docker-compose[\w.-]*\.ya?ml|requirements[\w.-]*\.(?:txt|lock|in)"
    r"|Dockerfile[\w.-]*|\.env\.[\w.-]+"
)
# Placeholders (`tests/test_<area>.py`), globs, shell variables and commands.
NOT_A_PATH = re.compile(r"[<>*{}$\s]")
# A word of code ends at whitespace, a quote, a bracket, a comma or "=", so the
# path in `pytest tests/test_api.py -q` or `open("app/main.py")` is read alone.
CODE_WORD = re.compile(r"[\s\"'(),=\[\]]+")


def _repo_path(token: str):
    """The repo-relative path a word of code names, or None if it names none.

    A location suffix (`:42`, `::test_name`, `:symbol`, `#section`) and
    surrounding punctuation are dropped: no file name here contains a colon,
    which Windows forbids anyway.
    """
    if not token or NOT_A_PATH.search(token):
        return None
    path = re.split(r"[:#]", token, maxsplit=1)[0]
    path = path.rstrip(".,;!?)'\"").lstrip("('\"").removeprefix("./")
    if path in ROOT_FILES or ROOT_FAMILIES.fullmatch(path):
        return path
    if "/" in path and path.split("/", 1)[0] in REPO_DIRS:
        return path
    return None


def _missing_paths(text: str) -> list:
    """(number, word) for each repo path in the code of `text` that does not
    exist: code spans and fenced blocks, word by word, so a renamed test file
    in `pytest tests/test_<area>.py -q` is caught too."""
    return [
        (number, word)
        for number, code in _code(text)
        for word in CODE_WORD.split(code)
        if (path := _repo_path(word)) and not _exists(ROOT / path)
    ]


@pytest.mark.parametrize("doc", LINKING_DOCS, ids=_rel)
def test_relative_links_resolve(doc):
    """Links and anchors are resolved the way GitHub renders them, exact case
    included; a link to a section that was renamed or never written fails."""
    broken = [
        f"{_rel(doc)}:{number}: {problem}" for number, problem in _broken_links(doc, _read(doc))
    ]
    assert not broken, "broken links:\n" + "\n".join(broken)


@pytest.mark.parametrize("doc", AGENT_DOCS, ids=_rel)
def test_backticked_repo_paths_exist(doc):
    """Example paths take a placeholder (`tests/test_<area>.py`), never an
    invented name an agent would go looking for. Commands count: every word
    of a code span or a fenced block is read."""
    missing = [f"{_rel(doc)}:{number}: `{token}`" for number, token in _missing_paths(_read(doc))]
    assert not missing, (
        "paths in code that do not exist (fix the path, or write a <placeholder>):\n"
        + "\n".join(missing)
    )


# "see KNOWN_ISSUES.md, section Web UI": the capitalised words after it, quoted
# or linked or not, start with a section heading. "section <name>" is skipped.
SECTION_CITATION = re.compile(
    r"KNOWN_ISSUES\.md[`*]*,? sections? [\"'*`\[]*([A-Z][^\"'*`\]]{0,79})"
)


def _known_issues_sections() -> list:
    path = ROOT / "KNOWN_ISSUES.md"
    if not _exists(path):
        return []
    return [
        heading.group(2) or ""
        for _, line, is_code in _lines(_read(path))
        if not is_code and (heading := HEADING.fullmatch(line)) and len(heading.group(1)) == 2
    ]


def _section_citations(text: str):
    """(number, words) after each "KNOWN_ISSUES.md, section" in the prose of `text`."""
    for start, block in _blocks(text):
        for match in SECTION_CITATION.finditer(" ".join(block.split())):
            yield start, match.group(1)


def test_known_issues_sections_cited_by_name_exist():
    """Docs point at a known issue as "see KNOWN_ISSUES.md, section <name>",
    not by a copy or an invented anchor, so a renamed section strands them."""
    sections = _known_issues_sections()
    stale = [
        f"{_rel(doc)}:{number}: section {cited.strip()!r}"
        for doc in LINKING_DOCS
        for number, cited in _section_citations(_read(doc))
        if not any(cited.startswith(name) for name in sections)
    ]
    assert not stale, (
        f"citations that start with none of the KNOWN_ISSUES.md sections {sections}:\n"
        + "\n".join(stale)
    )


# =========================
# Commands in the agent docs
# =========================

# python, python3, python3.10, python.exe, or the Windows launcher `py -3.10`,
# then interpreter options (`-X utf8`, `-u`), then the module. The optional
# second group catches a placeholder or glob right after it (`scripts.<name>`,
# `evaluation.*`), which names no module.
PYTHON_M = re.compile(
    r"(?<![\w.-])(?:python(?:3(?:\.\d+)?)?(?:\.exe)?|py(?:\.exe)?(?:\s+-3(?:\.\d+)?)?)"
    r"(?:\s+-[XW]\s*\S+|\s+-[bBdEiIOPqsSuv]+)*"
    r"\s+-m\s+([\w.]+)([<>*{}$\[])?"
)
MODULE = re.compile(r"\w+(?:\.\w+)*")


def _our_packages() -> set:
    """Top-level packages, i.e. what `python -m` can run from the repo root."""
    return {
        item.name for item in ROOT.iterdir()
        if item.is_dir() and (item / "__init__.py").exists()
    }


def _shortened_modules(ours: set) -> dict:
    """{name: full name} for each module of ours written without its leading
    package(s), as in `python -m sweep` for scripts.sweep, which fails from the
    repo root. A name the standard library also has is left out."""
    shortened = {}
    for package in sorted(ours):
        for file in sorted((ROOT / package).rglob("*.py")):
            parts = file.relative_to(ROOT).with_suffix("").parts
            if parts[-1] in ("__init__", "__main__"):
                parts = parts[:-1]
            for start in range(1, len(parts)):
                name = ".".join(parts[start:])
                if name.split(".")[0] not in sys.stdlib_module_names:
                    shortened.setdefault(name, ".".join(parts))
    return shortened


def _python_m_modules(text: str) -> list:
    """(number, module) for every `python -m` in `text`, in prose, code spans
    and fenced code alike; a placeholder or glob is skipped."""
    found = []
    for start, chunk in [*_blocks(text), *_fenced_lines(text)]:
        for match in PYTHON_M.finditer(chunk):
            if not match.group(2):
                number = start + chunk.count("\n", 0, match.start())
                found.append((number, match.group(1).rstrip(".")))
    return sorted(found)


def _is_module(dotted: str) -> bool:
    if not MODULE.fullmatch(dotted):
        return False
    path = ROOT.joinpath(*dotted.split("."))
    return _exists(path.parent / f"{path.name}.py") or _exists(path / "__main__.py")


def test_every_python_m_command_is_a_real_module():
    """Same contract as for README.md and DOCUMENTATION.md: a copy-pasteable
    command that does not exist reads as a promise that the tooling is there.
    Other tools (`python -m pytest`) are not this repository's to pin."""
    ours = _our_packages()
    shortened = _shortened_modules(ours)
    found = [
        (doc, number, module)
        for doc in AGENT_DOCS
        for number, module in _python_m_modules(_read(doc))
        if module.split(".")[0] in ours or module in shortened
    ]
    assert found, "no `python -m` command of ours in the agent docs, so this proves nothing"

    broken = [
        f"{_rel(doc)}:{number}: python -m {module}"
        + (f" (did you mean python -m {shortened[module]}?)" if module in shortened else "")
        for doc, number, module in found
        if not _is_module(module)
    ]
    assert not broken, "commands that name no module of ours:\n" + "\n".join(broken)


RUFF_CHECK = re.compile(r"\bruff(?:\.exe)?\s+check\b([^`#;&|()\n]*)")
# `ruff check` options whose value is the next argument (`ruff check --help`).
RUFF_VALUE_OPTIONS = {
    "-o", "--output-file", "--output-format", "--target-version", "--extension",
    "--select", "--ignore", "--extend-select", "--per-file-ignores",
    "--extend-per-file-ignores", "--fixable", "--unfixable", "--extend-fixable",
    "--exclude", "--extend-exclude", "--cache-dir", "--stdin-filename", "--config",
}
BARE_NAME = re.compile(r"[\w-]+")


def _lint_targets(args: list) -> list:
    """The paths among `ruff check` arguments: options and their values dropped."""
    targets, takes_value = [], False
    for arg in args:
        if takes_value:
            takes_value = False
        elif arg.startswith("-"):
            takes_value = arg in RUFF_VALUE_OPTIONS
        else:
            targets.append(arg)
    return targets


def _full_lint_commands(text: str) -> list:
    """(number, targets) for each `ruff check` in code whose targets are all bare
    top-level names: the lint-everything command, which must match CI.

    `ruff check app/rag/chain.py` or `ruff check .` is a different command and
    is not compared, and neither is "ruff check" in running prose, where the
    next words are not arguments. `app/` and `./app` count as `app`.
    """
    found = []
    for number, code in _code(text):
        for match in RUFF_CHECK.finditer(code):
            args = _lint_targets(match.group(1).split())
            targets = [arg.removeprefix("./").rstrip("/") for arg in args]
            if targets and all(BARE_NAME.fullmatch(arg) for arg in targets):
                found.append((number, targets))
    return found


def _ci_lint_targets() -> list:
    """What CI lints, read the way test_docs_consistency.py reads it."""
    workflow = _read(ROOT / ".github" / "workflows" / "ci.yml")
    match = re.search(r"ruff check ([\w /]+)", workflow)
    assert match, "no `ruff check` command in .github/workflows/ci.yml"
    return match.group(1).split()


def test_every_full_lint_command_matches_ci():
    """The package list is copied by hand into every doc that shows how to
    lint, and README.md and DOCUMENTATION.md once lagged behind CI."""
    expected = _ci_lint_targets()
    docs = [ROOT / "README.md", ROOT / "DOCUMENTATION.md", *AGENT_DOCS]

    wrong = []
    for doc in docs:
        for number, targets in _full_lint_commands(_read(doc)):
            missing = [name for name in expected if name not in targets]
            extra = [name for name in targets if name not in expected]
            if missing or extra:
                wrong.append(
                    f"{_rel(doc)}:{number}: ruff check {' '.join(targets)}"
                    f" (missing {missing}, not in CI {extra})"
                )
    assert not wrong, (
        f"lint commands that differ from CI's `ruff check {' '.join(expected)}`:\n"
        + "\n".join(wrong)
    )


@pytest.mark.parametrize("name", ["README.md", "DOCUMENTATION.md"])
def test_human_docs_show_the_full_lint_command(name):
    assert _full_lint_commands(_read(ROOT / name)), (
        f"{name} shows no `ruff check <packages>` command, so a reader cannot lint like CI"
    )


# =========================
# Machine-specific paths
# =========================

# Home directories, WSL drive mounts, Windows user profiles, and the per-session
# scratch directory Claude Code works in.
MACHINE_PATH = re.compile(
    r"/Users/|/home/|/mnt/[A-Za-z]/|(?i:\b[a-z]:[\\/]+users\b)|/tmp/claude-\d+/"
)


def _machine_paths(text: str) -> list:
    """(number, match, line) for each absolute path that only exists on one machine."""
    return [
        (number, match.group(0), line.strip())
        for number, line in enumerate(text.splitlines(), start=1)
        for match in MACHINE_PATH.finditer(line)
    ]


@pytest.mark.parametrize("doc", AGENT_DOCS, ids=_rel)
def test_no_machine_specific_paths(doc):
    """A home directory or a mounted drive is true on one machine only; write
    repo-relative paths, or generic forms such as `.venv/Scripts/python.exe`."""
    found = [
        f"{_rel(doc)}:{number}: {match!r} in {line[:120]!r}"
        for number, match, line in _machine_paths(_read(doc))
    ]
    assert not found, "machine-specific paths:\n" + "\n".join(found)
