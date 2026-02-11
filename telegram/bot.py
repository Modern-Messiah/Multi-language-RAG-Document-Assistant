import os
import logging
import httpx
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
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

LANGUAGES = [
    "Auto", "English", "Русский", "Қазақша", 
    "Français", "Deutsch", "Español", "中文", "日本語"
]

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
                params={"user_id": user_id}
            )
            if response.status_code == 200:
                await update.message.reply_text("✅ All your documents have been cleared!")
            else:
                await update.message.reply_text("❌ Failed to clear documents.")
    except Exception as e:
        logger.error(f"Error in clear command: {e}")
        await update.message.reply_text("❌ An error occurred.")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle uploaded documents."""
    doc = update.message.document
    file_name = doc.file_name
    
    if not file_name.lower().endswith(('.pdf', '.txt')):
        await update.message.reply_text("Sorry, I only support PDF and TXT files.")
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
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                await status_msg.edit_text(
                    f"✅ Document processed successfully!"
                )
            else:
                detail = response.json().get('detail', 'Unknown error')
                await status_msg.edit_text(f"❌ Failed to process document: {detail}")
                
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
            response = await client.post(f"{BACKEND_URL}/query", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer found.")
                sources = data.get("sources", [])
                
                # Format response
                msg = f"<b>Answer:</b>\n{answer}"
                
                if sources:
                    msg += "\n\n<b>Sources:</b>"
                    for src in sources:
                        msg += f"\n• <i>{src['source']}</i>"
                
                await update.message.reply_html(msg)
            else:
                detail = response.json().get('detail', 'No documents uploaded yet')
                await update.message.reply_text(f"❌ {detail}")
                
    except Exception as e:
        logger.error(f"Error handling query: {e}")
        await update.message.reply_text("❌ An error occurred while processing your question.")

def main():
    """Start the bot."""
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not found in environment variables!")
        return

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("clear", clear_command))
    
    # Handle documents
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    
    # Handle text messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot started...")
    app.run_polling()

if __name__ == '__main__':
    main()
