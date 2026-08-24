import html
import logging
import os

import httpx
from dotenv import load_dotenv
from telegram import KeyboardButton, ReplyKeyboardMarkup, Update
from telegram.constants import FileSizeLimit
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
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
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Shared secret for the backend; empty means the backend runs with auth
# disabled (development mode) and the header is simply ignored.
BACKEND_HEADERS = {"X-API-Key": os.getenv("BACKEND_API_KEY", "")}

LANGUAGES = [
    "Auto", "English", "Русский", "Қазақша", 
    "Français", "Deutsch", "Español", "中文", "日本語"
]

# Copying .env.template without editing it leaves these in place, and because
# they are non-empty the bot used to sail past its own "token missing" check
# and die inside python-telegram-bot with an opaque InvalidToken.
PLACEHOLDER_TOKENS = frozenset({
    "your-telegram-bot-token-here",
    "your-token-here",
})


def backend_error(response) -> str:
    """Extract a safe, human-readable reason from a failed backend response.

    Two things this must not do: raise (response.json() throws on a non-JSON
    body such as a proxy's HTML 502 page or an empty 500), and relay a
    configuration problem to an end user who cannot act on it.
    """
    if response.status_code in (401, 403):
        # "Invalid or missing API key" is for the operator, not the user.
        logger.error(
            "Backend rejected the bot's API key (HTTP %s) — check BACKEND_API_KEY",
            response.status_code,
        )
        return "I am not configured correctly. Please contact the operator."

    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    return str(detail) if detail else f"Backend error (HTTP {response.status_code})"


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
        "4. Use /clear to delete your uploaded documents.\n"
        "5. Use /start to reset conversation."
    )
    await update.message.reply_text(help_text)

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear user documents."""
    user_id = str(update.effective_user.id)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{BACKEND_URL}/clear",
                params={"user_id": user_id},
                headers=BACKEND_HEADERS
            )
            if response.status_code == 200:
                await update.message.reply_text("✅ All your documents have been cleared!")
            else:
                await update.message.reply_text(
                    f"❌ Failed to clear documents: {backend_error(response)}"
                )
    except Exception as e:
        logger.error(f"Error in clear command: {e}")
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
            f"That file is too big for me to download — Telegram limits bots "
            f"to {limit_mb} MB. Try splitting it."
        )
        return

    status_msg = await update.message.reply_text(f"Processing {file_name}...")
    
    try:
        # Get file from telegram
        tg_file = await context.bot.get_file(doc.file_id)
        file_bytes = await tg_file.download_as_bytearray()
        
        # Upload to FastAPI backend
        async with httpx.AsyncClient(timeout=60.0) as client:
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
    language = context.user_data.get("language", "Auto")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "question": text,
                "language": language,
                "user_id": str(update.effective_user.id)
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
                # with "can't parse entities".
                msg = f"<b>Answer:</b>\n{html.escape(answer)}"

                if sources:
                    msg += "\n\n<b>Sources:</b>"
                    for src in sources:
                        name = html.escape(str(src.get("source", "unknown")))
                        msg += f"\n• <i>{name}</i>"
                
                await update.message.reply_html(msg)
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
            "TELEGRAM_BOT_TOKEN is still the .env.template placeholder — "
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

    # Handle documents
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # Handle text messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.add_error_handler(on_error)

    logger.info("Bot started...")
    app.run_polling()

if __name__ == '__main__':
    main()
