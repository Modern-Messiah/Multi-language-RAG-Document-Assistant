"""Client-side code: the Streamlit UI and the Telegram bot.

A real package rather than loose scripts, for two reasons. The bot used to live
in a directory called `telegram/`, which collides with the installed
python-telegram-bot package - harmless only because it had no __init__.py, and
already enough to make two ruff versions disagree about how to sort its
imports. And a package can be imported by tests; a script that calls
load_dotenv() and builds an Application at module scope cannot.
"""
