"""One language table, and everything else derived from it.

The list used to exist three times over - prompt rules in the chain, a keyboard
in the bot, a radio group in the sidebar - with nothing keeping them in step. A
client offering a language the chain has no rule for silently ignores the user's
choice, which looks like the model disobeying rather than a config bug.

These tests assert the derivation, not a fourth copy of the list.
"""
import ast
from pathlib import Path

import pytest

from app.rag import chain, languages
from clients import backend

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend" / "streamlit_app.py"


def _module_literal(path: Path, name: str):
    """Read a literal assignment out of a module without importing it.

    streamlit_app.py calls st.set_page_config() at import time, so it cannot be
    imported in a test process.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path.name}")


# =========================
# The single source
# =========================

def test_supported_languages_is_auto_plus_every_rule():
    assert languages.SUPPORTED_LANGUAGES[0] == languages.AUTO_LANGUAGE
    assert set(languages.SUPPORTED_LANGUAGES[1:]) == set(languages.LANG_RULES)


def test_auto_is_not_itself_a_rule():
    """Auto means "mirror the question", so it must not sit in the rule table."""
    assert languages.AUTO_LANGUAGE not in languages.LANG_RULES


def test_no_duplicates():
    assert len(set(languages.SUPPORTED_LANGUAGES)) == len(languages.SUPPORTED_LANGUAGES)


def test_every_rule_is_a_non_empty_instruction():
    for name, rule in languages.LANG_RULES.items():
        assert rule.strip(), f"{name} has an empty rule"


# =========================
# rule_for
# =========================

def test_rule_for_returns_the_declared_rule():
    for name, rule in languages.LANG_RULES.items():
        assert languages.rule_for(name) == rule


def test_rule_for_auto_asks_to_mirror_the_question():
    assert languages.rule_for(languages.AUTO_LANGUAGE) == languages.AUTO_RULE


@pytest.mark.parametrize("unknown", ["Klingon", "", "en", "русский"])
def test_rule_for_unknown_language_falls_back_to_auto(unknown):
    """A client may be older than the backend; that must not be an error."""
    assert languages.rule_for(unknown) == languages.AUTO_RULE


# =========================
# Everything else derives from it
# =========================

def test_chain_re_exports_the_same_table():
    assert chain.LANG_RULES is languages.LANG_RULES


def test_shared_client_module_re_exports_the_same_list():
    assert backend.SUPPORTED_LANGUAGES is languages.SUPPORTED_LANGUAGES
    assert backend.AUTO_LANGUAGE == languages.AUTO_LANGUAGE


def test_bot_keyboard_offers_exactly_the_supported_languages():
    from clients import telegram_bot

    assert list(telegram_bot.LANGUAGES) == list(languages.SUPPORTED_LANGUAGES)


def test_bot_keyboard_is_laid_out_three_per_row():
    from clients import telegram_bot

    markup = telegram_bot.get_language_keyboard()
    rows = markup.keyboard

    flattened = [button.text for row in rows for button in row]
    assert flattened == list(languages.SUPPORTED_LANGUAGES)
    assert all(len(row) <= 3 for row in rows)


def test_frontend_derives_its_picker_instead_of_hardcoding_it():
    source = FRONTEND.read_text(encoding="utf-8")

    assert "SUPPORTED_LANGUAGES" in source, (
        "the sidebar picker must be built from the shared list, not a fourth copy"
    )


def test_frontend_has_a_flag_for_every_supported_language():
    flags = _module_literal(FRONTEND, "LANG_FLAGS")

    missing = [name for name in languages.SUPPORTED_LANGUAGES if name not in flags]
    assert not missing, f"languages with no flag in the sidebar: {missing}"


def test_frontend_flags_name_no_unknown_language():
    flags = _module_literal(FRONTEND, "LANG_FLAGS")

    unknown = sorted(set(flags) - set(languages.SUPPORTED_LANGUAGES))
    assert not unknown, f"sidebar flags for languages that do not exist: {unknown}"
