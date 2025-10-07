from duckduckgo_search import DDGS
from typing import List
import re


TOPIC_QUESTIONS = {
    "java": [
        "What is Java and why is it used?",
        "Is Java a coffee?",
        "What does slang Java mean?",
        "Do I need Java on my computer?",
        "What is Java programming language?",
        "How does Java work?",
        "Why use Java?",
        "When to use Java?",
        "Where is Java used?",
        "Who created Java?"
    ],
    "python": [
        "What is Python programming?",
        "How does Python work?",
        "Why use Python?",
        "When to use Python?",
        "Where is Python used?",
        "Who created Python?",
        "Which Python version is best?",
        "Is Python good for beginners?",
        "Should I learn Python?",
        "What can Python do?"
    ],
    "javascript": [
        "What is JavaScript?",
        "How does JavaScript work?",
        "Why use JavaScript?",
        "When to use JavaScript?",
        "Where is JavaScript used?",
        "Who created JavaScript?",
        "Which JavaScript framework is best?",
        "Is JavaScript hard to learn?",
        "Should I learn JavaScript?",
        "What can JavaScript do?"
    ],
    "tcp": [
        "What is TCP protocol?",
        "How does TCP work?",
        "Why use TCP?",
        "When to use TCP?",
        "Where is TCP used?",
        "Who created TCP?",
        "Which is better TCP or UDP?",
        "Is TCP reliable?",
        "Should I use TCP?",
        "What does TCP do?"
    ],
    "udp": [
        "What is UDP protocol?",
        "How does UDP work?",
        "Why use UDP?",
        "When to use UDP?",
        "Where is UDP used?",
        "Who created UDP?",
        "Which is better UDP or TCP?",
        "Is UDP faster than TCP?",
        "Should I use UDP?",
        "What does UDP do?"
    ],
    "tcp": [
        "What is TCP protocol?",
        "How does TCP work?",
        "Why use TCP?",
        "When to use TCP?",
        "Where is TCP used?",
        "Who created TCP?",
        "Which is better TCP or UDP?",
        "Is TCP reliable?",
        "Should I use TCP?",
        "What does TCP do?"
    ],
    "javascript": [
        "What is JavaScript?",
        "How does JavaScript work?",
        "Why use JavaScript?",
        "When to use JavaScript?",
        "Where is JavaScript used?",
        "Who created JavaScript?",
        "Which JavaScript framework is best?",
        "Is JavaScript hard to learn?",
        "Should I learn JavaScript?",
        "What can JavaScript do?"
    ]
}


def ddg_related(q: str, limit: int = 8) -> List[str]:
    words = q.lower().split()
    main_topic = None
    for w in words:
        if len(w) >= 3 and w not in ["about", "tell", "me", "what", "is", "how", "why", "when", "where"]:
            main_topic = w
            break

    if main_topic and main_topic in TOPIC_QUESTIONS:
        return TOPIC_QUESTIONS[main_topic][:limit]

    out = []

    with DDGS() as ddgs:
        try:
            for s in ddgs.suggestions(q):
                t = s.get("phrase") or s.get("title") or ""
                if t and len(t) > 3:
                    out.append(t)
        except:
            pass

        if main_topic:
            variations = [
                main_topic,
                f"what is {main_topic}",
                f"how to {main_topic}",
                f"{main_topic} tutorial",
                f"{main_topic} guide"
            ]

            for var in variations[:3]:
                try:
                    for s in ddgs.suggestions(var):
                        t = s.get("phrase") or s.get("title") or ""
                        if t and len(t) > 3:
                            out.append(t)
                except:
                    pass

    seen, qs = set(), []
    for s in out:
        s = s.strip()
        if not s or s.lower() in seen or len(s) < 5:
            continue
        seen.add(s.lower())

        if "?" in s or re.match(r"^(what|how|why|when|where|who|which|is|are|can|does|did|should|will|would|could)\b", s, re.I):
            if not s.endswith("?"):
                s = s + "?"
            qs.append(s)
        if len(qs) >= limit:
            break

    if len(qs) < limit and main_topic:
        smart_questions = [
            f"What is {main_topic}?",
            f"How does {main_topic} work?",
            f"Why use {main_topic}?",
            f"When to use {main_topic}?",
            f"Where is {main_topic} used?",
            f"Who created {main_topic}?",
            f"Which {main_topic} is best?",
            f"Is {main_topic} good?",
            f"Should I learn {main_topic}?",
            f"Will {main_topic} help me?",
            f"Does {main_topic} work?",
            f"Can I use {main_topic}?",
            f"What does {main_topic} do?",
            f"How to learn {main_topic}?",
            f"Why is {main_topic} popular?"
        ]

        for sq in smart_questions:
            if sq.lower() not in seen and len(qs) < limit:
                seen.add(sq.lower())
                qs.append(sq)

    return qs[:limit]
