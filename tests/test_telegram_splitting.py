"""Splitting long answers under Telegram's message limit.

The bot sent every answer as one reply_html. Past 4096 characters Telegram
answers BadRequest, the generic handler logged it and told the user "an error
occurred" - so a question that produced a long, correct answer looked like a
crash.
"""
import html

import pytest
from telegram.constants import MessageLimit

from clients.telegram_bot import _CHUNK_BUDGET, split_for_telegram

LIMIT = MessageLimit.MAX_TEXT_LENGTH


def _fits(chunk: str) -> bool:
    """What the bot actually sends is the escaped chunk under a heading."""
    rendered = f"<b>Answer (9/9):</b>\n{html.escape(chunk)}"
    return len(rendered) <= LIMIT


# =========================
# Short input is left alone
# =========================

@pytest.mark.parametrize("text", ["", "short answer", "line\nline\n", "  "])
def test_short_text_is_one_chunk(text):
    assert split_for_telegram(text) == [text]


def test_text_exactly_at_the_budget_is_not_split():
    text = "a" * _CHUNK_BUDGET

    assert split_for_telegram(text) == [text]


# =========================
# Long input is split, and every piece fits
# =========================

def test_long_text_is_split():
    text = "word " * 3000

    chunks = split_for_telegram(text)

    assert len(chunks) > 1


@pytest.mark.parametrize(
    "text",
    [
        "word " * 3000,
        "x" * 20000,
        ("paragraph text here.\n\n" * 800),
        "线" * 5000,
    ],
    ids=["words", "no-separators", "paragraphs", "cjk"],
)
def test_every_chunk_fits_a_telegram_message(text):
    for chunk in split_for_telegram(text):
        assert _fits(chunk), f"chunk of {len(chunk)} chars does not fit once escaped"


def test_nothing_is_lost_or_duplicated():
    text = "word " * 3000

    assert "".join(split_for_telegram(text)) == text


def test_order_is_preserved():
    text = "\n".join(f"line {i}" for i in range(4000))

    joined = "".join(split_for_telegram(text))

    assert joined == text


# =========================
# The escaping trap
# =========================

def test_markup_heavy_text_is_measured_after_escaping():
    """"<" becomes "&lt;" - four characters for one, so a chunk sized on the
    raw text would overflow the moment it is escaped."""
    text = "<" * 3000

    chunks = split_for_telegram(text)

    assert len(chunks) > 1
    for chunk in chunks:
        assert _fits(chunk)


def test_entities_are_never_cut_in_half():
    """Splitting after escaping could leave a chunk ending in "&l"."""
    text = "<tag> & more " * 1000

    for chunk in split_for_telegram(text):
        escaped = html.escape(chunk)
        # An escaped chunk must not end mid-entity.
        tail = escaped[-8:]
        assert "&" not in tail or ";" in tail.split("&")[-1], f"cut entity: {tail!r}"


# =========================
# Boundary preference
# =========================

def test_split_prefers_paragraph_boundaries():
    paragraph = "This is a sentence that fills space. " * 40  # ~1480 chars
    text = "\n\n".join([paragraph] * 6)

    chunks = split_for_telegram(text)

    assert len(chunks) > 1
    # At least one break should land right after a blank line.
    assert any(chunk.endswith("\n\n") for chunk in chunks[:-1])


def test_a_wall_of_text_still_makes_progress():
    """No separators anywhere: the splitter must not loop or emit empties."""
    text = "x" * 20000

    chunks = split_for_telegram(text)

    assert len(chunks) > 1
    assert all(chunk for chunk in chunks), "produced an empty chunk"
    assert "".join(chunks) == text


def test_chunk_count_is_reasonable():
    """A greedy splitter should not shred text into tiny pieces."""
    text = "word " * 3000  # 15000 chars

    chunks = split_for_telegram(text)

    assert len(chunks) <= 6, f"{len(chunks)} chunks for 15k chars is too fragmented"


def test_a_custom_budget_is_respected():
    text = "word " * 200

    for chunk in split_for_telegram(text, budget=100):
        assert len(html.escape(chunk)) <= 100
