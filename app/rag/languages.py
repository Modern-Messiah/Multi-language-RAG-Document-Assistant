"""The supported answer languages, in one place.

This list used to exist three times: as prompt rules in the chain, as a
keyboard in the Telegram bot, and as a radio group in the Streamlit sidebar.
Nothing kept them in step, and a client offering a language the chain has no
rule for silently ignores the user's choice.

Deliberately free of heavy imports so both clients can read it without pulling
in openai, langchain or chromadb.
"""

# "Auto" is not a rule: it tells the model to mirror the question's language.
AUTO_LANGUAGE = "Auto"

AUTO_RULE = (
    "Answer in the same language as the user's question. "
    "If the context is in another language, translate the answer."
)

LANG_RULES = {
    "English": "Answer strictly in English.",
    "Русский": "Отвечай строго на русском языке.",
    "Қазақша": "Жауапты қатаң түрде қазақ тілінде бер.",
    "Français": "Réponds strictement en français.",
    "Deutsch": "Antworte ausschließlich auf Deutsch.",
    "Español": "Responde estrictamente en español.",
    "中文": "请严格使用简体中文回答。",
    "日本語": "必ず日本語で回答してください。",
}

# What a client may offer: Auto first, then every language with a rule.
SUPPORTED_LANGUAGES = (AUTO_LANGUAGE, *LANG_RULES)


def rule_for(language: str) -> str:
    """The prompt rule for a language, falling back to mirroring the question.

    An unknown value is not an error: a client may be older than the backend,
    and answering in the question's language is the sane default.
    """
    if language == AUTO_LANGUAGE:
        return AUTO_RULE
    return LANG_RULES.get(language, AUTO_RULE)
