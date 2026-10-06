import re

_TOPIC_PATTERNS = [
    re.compile(r"what don'?t i know about (.+?)\??$", re.I),
    re.compile(r"what do i not know about (.+?)\??$", re.I),
    re.compile(r"knowledge gaps?(?: in| on| about) (.+?)\??$", re.I),
    re.compile(r"what am i missing (?:on|about) (.+?)\??$", re.I),
    re.compile(r"gaps? in my (?:knowledge|understanding) (?:of|on|about) (.+?)\??$", re.I),
    re.compile(r"about (.+?)\??$", re.I),
]


def extract_focus_topic(query: str) -> str:
    text = query.strip()
    for pattern in _TOPIC_PATTERNS:
        match = pattern.search(text)
        if match:
            topic = match.group(1).strip(" .?!\"'")
            if topic:
                return topic
    return text
