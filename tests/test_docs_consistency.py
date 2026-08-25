"""Documentation must not drift from the code again.

Stage 2 inherited docs that described optional user_id, a `streamlit_user`
tenant, 500/50 chunking and no auth header - all false. These tests pin the
mechanical parts of that contract so the next drift fails CI instead of
misleading a reader.
"""
import re
from pathlib import Path

import pytest

from app.config import Settings

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTATION = (ROOT / "DOCUMENTATION.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
ENV_TEMPLATE = (ROOT / ".env.template").read_text(encoding="utf-8")

# Documented in the config table but read directly by the clients, not Settings.
CLIENT_ONLY_VARS = {"TELEGRAM_BOT_TOKEN", "BACKEND_URL"}


def _table_rows():
    """{VAR: default cell} from the Configuration table."""
    return dict(re.findall(r"^\| `([A-Z_]+)` \|[^|]*\| (.+?) \|$", DOCUMENTATION, re.M))


def _env_template_vars():
    return set(re.findall(r"^([A-Z_]+)=", ENV_TEMPLATE, re.M))


def _normalise(value: str) -> str:
    """Compare paths without caring about separator or a leading './'."""
    return value.replace("\\", "/").lstrip("./").rstrip("/")


def _settings_defaults():
    settings = Settings(_env_file=None, openai_api_key="placeholder")
    return {name.upper(): getattr(settings, name) for name in Settings.model_fields}


# =========================
# Configuration table
# =========================

def test_every_setting_appears_in_the_config_table():
    documented = set(_table_rows())

    missing = sorted(set(_settings_defaults()) - documented)
    assert not missing, f"settings missing from the Configuration table: {missing}"


def test_config_table_has_no_invented_variables():
    known = set(_settings_defaults()) | CLIENT_ONLY_VARS

    invented = sorted(set(_table_rows()) - known)
    assert not invented, f"documented variables that do not exist: {invented}"


@pytest.mark.parametrize(
    "name",
    sorted(set(_settings_defaults()) - {"OPENAI_API_KEY"}),
)
def test_documented_default_matches_the_code(name):
    cell = _table_rows()[name]
    actual = _settings_defaults()[name]

    # Defaults are written as `value` in the table; an empty default as `""`.
    quoted = re.findall(r"`([^`]*)`", cell)
    assert quoted, f"{name}: no backticked default in {cell!r}"

    expected = "" if quoted[0] == '""' else quoted[0]
    assert _normalise(expected) == _normalise(str(actual)), (
        f"{name}: docs say {expected!r}, code default is {actual!r}"
    )


# =========================
# .env.template
# =========================

def test_env_template_covers_every_setting():
    missing = sorted(set(_settings_defaults()) - _env_template_vars())
    assert not missing, f".env.template is missing: {missing}"


def test_env_template_has_no_unknown_variables():
    known = set(_settings_defaults()) | CLIENT_ONLY_VARS

    unknown = sorted(_env_template_vars() - known)
    assert not unknown, f".env.template documents unknown variables: {unknown}"


# =========================
# Claims that were previously false
# =========================

def test_docs_do_not_claim_user_id_is_optional():
    assert "user_id` (optional)" not in DOCUMENTATION
    assert "user_id (optional)" not in DOCUMENTATION


def test_docs_do_not_mention_the_retired_streamlit_user_tenant():
    assert "streamlit_user" not in DOCUMENTATION
    assert "streamlit_user" not in README


def test_docs_describe_the_api_key_header():
    for text, label in ((DOCUMENTATION, "DOCUMENTATION.md"), (README, "README.md")):
        assert "BACKEND_API_KEY" in text, f"{label} never mentions BACKEND_API_KEY"
    assert "X-API-Key" in DOCUMENTATION


def test_docs_state_the_real_chunking_defaults():
    """The old text claimed 500/50; the values must come from the code."""
    settings = Settings(_env_file=None, openai_api_key="placeholder")

    chunking_lines = [
        line for line in DOCUMENTATION.splitlines()
        if "hunking" in line and "haracter" in line
    ]
    assert chunking_lines, "no line describes the chunking defaults"

    described = " ".join(chunking_lines)
    assert str(settings.chunk_size) in described, described
    assert str(settings.chunk_overlap) in described, described
    assert "500" not in described and "50 char" not in described


BOT_COMMANDS = {"start", "help", "clear", "documents"}


def _registered_bot_commands():
    bot_source = (ROOT / "clients" / "telegram_bot.py").read_text(encoding="utf-8")
    return set(re.findall(r'CommandHandler\("(\w+)"', bot_source))


def test_docs_do_not_advertise_nonexistent_bot_commands():
    for advertised in re.findall(r"`?/(\w+)`?", README):
        if advertised in {"upload", "query"}:
            pytest.fail(f"README advertises /{advertised}, which the bot does not handle")

    assert _registered_bot_commands() == BOT_COMMANDS


def test_every_bot_command_is_documented():
    """A command nobody knows about may as well not exist."""
    for command in _registered_bot_commands():
        assert f"/{command}" in README, f"/{command} is not in README.md"

    help_text = (ROOT / "clients" / "telegram_bot.py").read_text(encoding="utf-8")
    help_block = help_text.split("help_text = (")[1].split(")")[0]
    for command in _registered_bot_commands() - {"help"}:
        assert f"/{command}" in help_block, f"/{command} is missing from /help"


def test_docs_reference_the_current_bot_entry_point():
    """The bot moved out of telegram/, which shadowed the installed package."""
    for text, label in ((README, "README.md"), (DOCUMENTATION, "DOCUMENTATION.md")):
        assert "telegram/bot.py" not in text, f"{label} still names the old bot path"
        assert "clients.telegram_bot" in text, f"{label} does not show how to run the bot"

    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "clients.telegram_bot" in compose
    assert "telegram/bot.py" not in compose


def test_production_compose_command_is_documented():
    """`docker compose up` silently merges the dev override."""
    for text, label in ((README, "README.md"), (DOCUMENTATION, "DOCUMENTATION.md")):
        assert "-f docker-compose.yml" in text, (
            f"{label} does not show how to run the production configuration"
        )
