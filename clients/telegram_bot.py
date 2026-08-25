import html
import logging
import os

import httpx
from dotenv import load_dotenv
from telegram import KeyboardButton, ReplyKeyboardMarkup, Update
from telegram.constants import ChatAction, FileSizeLimit, MessageLimit
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from clients.backend import (
    AUTO_LANGUAGE,
    SUPPORTED_LANGUAGES,
    api_headers,
    backend_url,
    error_from_response,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BACKEND_URL = backend_url()

# Shared secret for the backend; empty means the backend runs with auth
# disabled (development mode) and the header is simply ignored.
BACKEND_HEADERS = api_headers()

# Auto plus every language the chain actually has a rule for.
LANGUAGES = list(SUPPORTED_LANGUAGES)

# The backend gives OpenAI 45 s and may retry, so a 60 s client timeout could
# abandon a request the server was still going to answer.
QUERY_TIMEOUT = 120.0
UPLOAD_TIMEOUT = 120.0
CLEAR_TIMEOUT = 30.0

# Exchanges kept per chat. The backend caps this again with MAX_HISTORY_TURNS;
# the bound here stops a long-running chat from growing the request body
# without limit.
HISTORY_TURNS = 6

# Copying .env.template without editing it leaves these in place, and because
# they are non-empty the bot used to sail past its own "token missing" check
# and die inside python-telegram-bot with an opaque InvalidToken.
PLACEHOLDER_TOKENS = frozenset({
    "your-telegram-bot-token-here",
    "your-token-here",
})


# Room inside a message for the "<b>Answer (2/3):</b>\n" heading.
_CHUNK_BUDGET = MessageLimit.MAX_TEXT_LENGTH - 200


def _escaped_length(text: str) -> int:
    return len(html.escape(text))


def split_for_telegram(text: str, budget: int = _CHUNK_BUDGET) -> list:
    """Cut raw text into pieces whose HTML-escaped form fits one message.

    Telegram rejects anything past MAX_TEXT_LENGTH, and the bot used to send
    the whole answer in a single reply_html: a long answer raised BadRequest,
    the generic handler swallowed it, and the user paid for an answer they
    never saw.

    Two subtleties drive the implementation. Escaping can quadruple a
    character ("<" becomes "&lt;"), so the budget has to be measured on the
    escaped form, not the raw one. And splitting AFTER escaping could cut an
    entity in half, so the raw text is what gets divided.
    """
    text = text or ""
    if _escaped_length(text) <= budget:
        return [text]

    chunks = []
    remaining = text
    while remaining:
        if _escaped_length(remaining) <= budget:
            chunks.append(remaining)
            break

        # Largest prefix that still fits once escaped.
        low, high = 1, len(remaining)
        while low < high:
            middle = (low + high + 1) // 2
            if _escaped_length(remaining[:middle]) <= budget:
                low = middle
            else:
                high = middle - 1
        cut = low

        # Prefer a natural boundary, but never give back more than half the
        # chunk chasing one - a wall of text with no separators must still
        # make progress.
        window = remaining[:cut]
        for separator in ("\n\n", "\n", ". ", " "):
            index = window.rfind(separator)
            if index > cut // 2:
                cut = index + len(separator)
                break

        chunks.append(remaining[:cut])
        remaining = remaining[cut:]

    return chunks


def format_document_list(documents: list) -> str:
    """Render the inventory as one HTML message.

    Pure on purpose: the bot had no way to show what it had indexed, and this
    formatting is the part worth testing without a Telegram server in the loop.
    """
    if not documents:
        return "You have no documents indexed. Send me a PDF or TXT file."

    lines = [f"<b>Your documents ({len(documents)}):</b>"]
    for doc in documents:
        name = html.escape(str(doc.get("source", "unknown")))
        chunks = doc.get("chunks", 0)
        detail = f"{chunks} chunk{'s' if chunks != 1 else ''}"
        if doc.get("pages"):
            pages = doc["pages"]
            detail += f", {pages} page{'s' if pages != 1 else ''}"
        lines.append(f"\n• <b>{name}</b>\n  <i>{detail}</i>")
    return "".join(lines)


def backend_error(response) -> str:
    """Describe a failed backend response, logging operator-only failures.

    The wording lives in clients.backend so the Streamlit UI says the same
    thing; the logging is the bot's own, since only the bot has an operator
    reading its log.
    """
    if response.status_code in (401, 403):
        logger.error(
            "Backend rejected the bot's API key (HTTP %s) - check BACKEND_API_KEY",
            response.status_code,
        )
    return error_from_response(response)


def get_language_keyboard():
    keyboard = []
    # 3 languages per row
    for i in range(0, len(LANGUAGES), 3):
        row = [KeyboardButton(lang) for lang in LANGUAGES[i:i+3]]
        keyboard.append(row)
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    user = update.effective_user
    context.user_data.pop("history", None)

    await update.message.reply_html(
        rf"Hi {user.mention_html()}! I am your RAG Document Assistant bot. "
        "\n\n<b>Please select your preferred answer language to continue:</b>"
        "\n\n<i>Use /clear to reset your documents.</i>",
        reply_markup=get_language_keyboard()
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    help_text = (
        "How to use me:\n"
        "1. Attach a PDF or TXT file and I will index it.\n"
        "2. Send any text message to ask questions about your documents.\n"
        "3. Select a language from the keyboard to set the response language.\n"
        "4. Use /documents to see what I have indexed.\n"
        "5. Use /clear to delete all of them.\n"
        "6. Use /start to reset conversation."
    )
    await update.message.reply_text(help_text)

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear user documents."""
    user_id = str(update.effective_user.id)
    try:
        async with httpx.AsyncClient(timeout=CLEAR_TIMEOUT) as client:
            response = await client.post(
                f"{BACKEND_URL}/clear",
                params={"user_id": user_id},
                headers=BACKEND_HEADERS
            )
            if response.status_code == 200:
                # The transcript refers to documents that no longer exist;
                # keeping it would feed the model a conversation about them.
                context.user_data.pop("history", None)
                await update.message.reply_text("✅ All your documents have been cleared!")
            else:
                await update.message.reply_text(
                    f"❌ Failed to clear documents: {backend_error(response)}"
                )
    except Exception as e:
        logger.error(f"Error in clear command: {e}")
        await update.message.reply_text("❌ An error occurred.")

async def documents_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List what this user has indexed.

    There was no way to ask before: the bot knew nothing about its own corpus,
    so the only way to deal with a stale file was /clear and start over.
    """
    user_id = str(update.effective_user.id)
    try:
        async with httpx.AsyncClient(timeout=CLEAR_TIMEOUT) as client:
            response = await client.get(
                f"{BACKEND_URL}/documents",
                params={"user_id": user_id},
                headers=BACKEND_HEADERS,
            )
            if response.status_code == 200:
                await update.message.reply_html(
                    format_document_list(response.json()["documents"])
                )
            else:
                await update.message.reply_text(f"❌ {backend_error(response)}")
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        await update.message.reply_text("❌ An error occurred.")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle uploaded documents."""
    doc = update.message.document
    # Document.file_name is Optional in the Bot API (some forwarded or
    # generated attachments arrive without one), so guard before .lower().
    file_name = doc.file_name or ""
    
    if not file_name.lower().endswith(('.pdf', '.txt')):
        await update.message.reply_text("Sorry, I only support PDF and TXT files.")
        return

    # Telegram refuses getFile above 20 MB regardless of what the backend
    # accepts, and the failure used to surface as a bare "an error occurred".
    if doc.file_size and doc.file_size > FileSizeLimit.FILESIZE_DOWNLOAD:
        limit_mb = FileSizeLimit.FILESIZE_DOWNLOAD // (1000 * 1000)
        await update.message.reply_text(
            f"That file is too big for me to download - Telegram limits bots "
            f"to {limit_mb} MB. Try splitting it."
        )
        return

    status_msg = await update.message.reply_text(f"Processing {file_name}...")
    
    try:
        await update.message.chat.send_action(ChatAction.TYPING)

        # Get file from telegram
        tg_file = await context.bot.get_file(doc.file_id)
        file_bytes = await tg_file.download_as_bytearray()
        
        # Upload to FastAPI backend
        async with httpx.AsyncClient(timeout=UPLOAD_TIMEOUT) as client:
            files = {'file': (file_name, bytes(file_bytes))}
            params = {'user_id': str(update.effective_user.id)}
            response = await client.post(
                f"{BACKEND_URL}/upload",
                files=files,
                params=params,
                headers=BACKEND_HEADERS
            )

            if response.status_code == 200:
                if response.json().get("duplicate"):
                    await status_msg.edit_text("ℹ️ This document is already indexed.")
                else:
                    await status_msg.edit_text("✅ Document processed successfully!")
            else:
                await status_msg.edit_text(
                    f"❌ Failed to process document: {backend_error(response)}"
                )
                
    except Exception as e:
        logger.error(f"Error handling document: {e}")
        await status_msg.edit_text("❌ An error occurred while processing the document.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages (questions or language selection)."""
    text = update.message.text
    
    # Check if this is a language selection
    if text in LANGUAGES:
        context.user_data["language"] = text
        await update.message.reply_text(
            f"✅ Language set to: {text}\n\n"
            "<b>Now, what would you like to do?</b>\n"
            "1. 📄 Attach a <b>PDF</b> or <b>TXT</b> file to index it.\n"
            "2. 💬 Ask me questions about your documents.\n\n"
            "You can change the language at any time by clicking the buttons below.",
            parse_mode='HTML'
        )
        return

    # Treat as query
    language = context.user_data.get("language", AUTO_LANGUAGE)
    
    try:
        # Retrieval plus generation takes seconds; without this the chat looks
        # like the bot ignored the question.
        await update.message.chat.send_action(ChatAction.TYPING)

        async with httpx.AsyncClient(timeout=QUERY_TIMEOUT) as client:
            history = context.user_data.get("history", [])
            payload = {
                "question": text,
                "language": language,
                "user_id": str(update.effective_user.id),
                "history": history[-HISTORY_TURNS:],
            }
            response = await client.post(
                f"{BACKEND_URL}/query", json=payload, headers=BACKEND_HEADERS
            )

            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer found.")
                sources = data.get("sources", [])

                # Escape before interpolating into HTML: both the answer and
                # the source names derive from user-supplied documents, and a
                # single stray "<" made Telegram reject the whole message
                # with "can't parse entities". Split first, escape after, so
                # an entity is never cut in half.
                parts = split_for_telegram(answer)
                for index, part in enumerate(parts, start=1):
                    heading = "Answer" if len(parts) == 1 else f"Answer ({index}/{len(parts)})"
                    await update.message.reply_html(
                        f"<b>{heading}:</b>\n{html.escape(part)}"
                    )

                if sources:
                    lines = ["<b>Sources:</b>"]
                    for src in sources:
                        name = html.escape(str(src.get("source", "unknown")))
                        lines.append(f"• <i>{name}</i>")
                    # Sent separately so a long answer cannot push the sources
                    # over the limit and lose them.
                    await update.message.reply_html("\n".join(lines))

                # Remember the exchange so the next question can be a follow-up.
                # Recorded only on success: storing a failed turn would teach
                # the model that an error message was a valid answer.
                history.append({"question": text, "answer": answer})
                context.user_data["history"] = history[-HISTORY_TURNS:]
            else:
                await update.message.reply_text(f"❌ {backend_error(response)}")

    except Exception as e:
        logger.error(f"Error handling query: {e}")
        await update.message.reply_text("❌ An error occurred while processing your question.")

async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Catch anything a handler let escape.

    Without a registered error handler python-telegram-bot only logs the
    exception, so the user is left staring at a message that never gets a
    reply.
    """
    logger.exception("Unhandled error while processing an update", exc_info=context.error)

    message = getattr(update, "effective_message", None)
    if message is not None:
        try:
            await message.reply_text("❌ Something went wrong on my side. Please try again.")
        except Exception:
            logger.exception("Could not deliver the error notice")


def main():
    """Start the bot."""
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not found in environment variables!")
        return

    if TELEGRAM_BOT_TOKEN in PLACEHOLDER_TOKENS:
        logger.error(
            "TELEGRAM_BOT_TOKEN is still the .env.template placeholder - "
            "put a real token from @BotFather in .env"
        )
        return

    # Without concurrent_updates the default processor handles one update at a
    # time, so a single 60-second document upload blocks every other user.
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .concurrent_updates(True)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("documents", documents_command))

    # Handle documents
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # Handle text messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.add_error_handler(on_error)

    logger.info("Bot started...")
    app.run_polling()

if __name__ == '__main__':
    main()
