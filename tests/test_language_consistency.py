"""The three language lists must not drift apart.

The chain, the Streamlit picker and the bot keyboard each hardcode their own
list. If a client offers a language the chain has no rule for, the request
still succeeds but silently ignores the user's choice - so compare them by
parsing the client sources (no imports, no side effects).
"""
import ast
from pathlib import Path

from app.rag.chain import LANG_RULES

ROOT = Path(__file__).resolve().parents[1]
AUTO = "Auto"


def _assigned_value(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path.name}")


def _bot_languages():
    return _assigned_value(ROOT / "telegram" / "bot.py", "LANGUAGES")


def _frontend_languages():
    # LANG_OPTIONS maps a decorated label ("English 🇬🇧") to the wire value.
    return list(_assigned_value(ROOT / "frontend" / "streamlit_app.py", "LANG_OPTIONS").values())


def test_bot_offers_auto_plus_every_chain_language():
    assert set(_bot_languages()) == {AUTO} | set(LANG_RULES)


def test_frontend_offers_auto_plus_every_chain_language():
    assert set(_frontend_languages()) == {AUTO} | set(LANG_RULES)


def test_clients_offer_the_same_languages():
    assert set(_bot_languages()) == set(_frontend_languages())


def test_no_client_language_is_missing_a_chain_rule():
    offered = (set(_bot_languages()) | set(_frontend_languages())) - {AUTO}

    missing = sorted(offered - set(LANG_RULES))
    assert not missing, f"clients offer languages the chain has no rule for: {missing}"


def test_language_lists_have_no_duplicates():
    bot = _bot_languages()
    assert len(bot) == len(set(bot)), "duplicate entry in the bot keyboard"
