"""A small labelled corpus and the questions it should answer.

Small on purpose. A golden set is only useful if someone maintains it, and a
hundred cases nobody re-reads rots into noise. These cover the properties worth
protecting: the right document is found, a near-miss document is not preferred
over the right one, multilingual questions work, and a question the corpus
cannot answer retrieves nothing worth using.
"""
from typing import Dict, List

# Documents are written so that topic overlap is visible in the words
# themselves, which is what makes the offline bag-of-words embedder in
# tests/test_retrieval_quality.py a fair stand-in for a real one.
CORPUS: Dict[str, str] = {
    "solar.txt": (
        "Solar panels convert sunlight into electricity using photovoltaic cells. "
        "A residential rooftop installation typically produces between four and "
        "eight kilowatts. Panel efficiency has improved steadily, and most modern "
        "photovoltaic modules convert around twenty percent of incoming sunlight."
    ),
    "wind.txt": (
        "Wind turbines generate electricity when moving air turns their blades. "
        "Offshore wind farms reach higher capacity factors than onshore ones "
        "because wind over open water is stronger and steadier. A single modern "
        "offshore turbine can exceed twelve megawatts of rated capacity."
    ),
    "battery.txt": (
        "Lithium ion batteries store energy chemically and release it on demand. "
        "Grid scale battery storage smooths the mismatch between solar generation "
        "during the day and household demand in the evening. Battery degradation "
        "depends mostly on temperature and depth of discharge."
    ),
    "recipe.txt": (
        "To bake sourdough bread, mix flour, water and starter, then let the dough "
        "ferment overnight. Shape the loaf, score the top and bake it in a very hot "
        "oven with steam for the first fifteen minutes."
    ),
    "отпуск.txt": (
        "Ежегодный оплачиваемый отпуск составляет двадцать восемь календарных дней. "
        "Отпуск предоставляется по заявлению сотрудника и согласуется с "
        "руководителем подразделения не позднее чем за две недели."
    ),
}


# expected is the set of sources a good retriever must surface for the question.
GOLDEN_CASES: List[Dict] = [
    {
        "question": "How efficient are photovoltaic modules?",
        "expected": ["solar.txt"],
        "note": "direct topical match",
    },
    {
        "question": "Why do offshore wind farms perform better?",
        "expected": ["wind.txt"],
        "note": "direct topical match, distinct vocabulary",
    },
    {
        "question": "What makes lithium ion batteries degrade?",
        "expected": ["battery.txt"],
        "note": "direct topical match",
    },
    {
        "question": "How is electricity generated from moving air?",
        "expected": ["wind.txt"],
        "note": "paraphrase - the document says 'wind turns their blades'",
    },
    {
        "question": "How do I bake sourdough?",
        "expected": ["recipe.txt"],
        "note": "off-topic document must still be findable on its own subject",
    },
    {
        "question": "Сколько дней длится ежегодный отпуск?",
        "expected": ["отпуск.txt"],
        "note": "Russian question against a Russian document",
    },
    {
        "question": "storage smooths solar generation against evening demand",
        "expected": ["battery.txt"],
        "note": "mentions solar but is about storage - must not prefer solar.txt",
    },
]

# Questions the corpus genuinely cannot answer. Retrieval will still return its
# nearest chunks; what matters is that they score low enough for a threshold to
# catch, which is what run_eval.py reports on.
UNANSWERABLE = [
    "What is the population of Ulaanbaatar?",
    "Как приготовить плов из баранины?",
]
