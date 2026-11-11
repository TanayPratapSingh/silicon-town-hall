"""Persona builder agent: a Census record becomes a short first person profile, then a stereotype review.

The review has two parts: keyword heuristics that always run, and an optional LLM reviewer. The
heuristics only catch wording on the list below; a profile that passes them can still lean on a
stereotype, which is why the LLM hook exists and why profiles still need a human read.
"""

import json
import re

PROFILE_FIELDS = ["age", "sex", "race_ethnicity", "income", "commute", "tenure", "disability", "neighborhood"]

PROFILE_SYSTEM = (
    "You write a short first person profile of a simulated resident for a public planning exercise. "
    "Use only the facts given. Write four to six plain sentences about the daily life those facts imply: "
    "getting around, housing costs, money, health and access. Do not invent a name, job, family, "
    "personality, accent, culture, religion or beliefs the facts do not state. Do not open with race, "
    "ethnicity or disability, and do not describe disability or low income as tragedy or inspiration."
)

REVIEW_SYSTEM = (
    "You review a simulated resident profile for stereotypes. Flag traits attributed to the person "
    "because of race, ethnicity, sex, age, income or disability that the profile's facts do not support; "
    "identity used as the opening or defining trait; and deficit or inspiration framing of disability or "
    'poverty. Reply as JSON: {"stereotype": true or false, "reasons": ["short reason", ...]}.'
)

# Phrases that usually signal a stereotype in a generated profile, with the kind of stereotype.
STEREOTYPE_TERMS = {
    "ghetto": "place or race", "the hood": "place or race", "thug": "race", "gang": "race or place",
    "crime ridden": "place", "dangerous neighborhood": "place", "welfare": "poverty", "handout": "poverty",
    "food stamps": "poverty", "illegal": "ethnicity or immigration", "broken english": "ethnicity",
    "exotic": "race or ethnicity", "articulate": "race", "spicy": "ethnicity", "fiery": "ethnicity or gender",
    "sassy": "race or gender", "feisty": "gender", "model minority": "race", "good at math": "race",
    "strong black woman": "race and gender", "hardworking immigrant": "ethnicity", "lazy": "race or poverty",
    "uneducated": "poverty", "wheelchair bound": "disability", "confined to a wheelchair": "disability",
    "suffers from": "disability", "suffering from": "disability", "afflicted": "disability",
    "despite my disability": "disability", "inspiration": "disability", "brave": "disability",
    "frail": "age", "senile": "age", "set in my ways": "age", "not good with technology": "age",
    "partying": "age", "bossy": "gender", "emotional": "gender",
}
IDENTITY_OPENER = re.compile(
    r"^\W*as an? (black|white|asian|hispanic|latina|latino|latinx|african american|native|indigenous|"
    r"puerto rican|mexican|dominican|disabled|immigrant|old|elderly|young|poor|low income)\b",
    re.IGNORECASE,
)


def facts(resident):
    """The fields the profile writer may use. Serial numbers and weights stay out of the prompt."""
    out = {key: resident.get(key) for key in PROFILE_FIELDS}
    out["annual_income_dollars"] = out.pop("income")  # a bare "income" was read as monthly in 1 of 20 real profiles
    return out


def build_profile(resident, llm):
    """A first person profile written by llm.chat from the resident's Census facts."""
    messages = [{"role": "system", "content": PROFILE_SYSTEM},
                {"role": "user", "content": "Facts (JSON): " + json.dumps(facts(resident))}]
    return llm.chat(messages, temperature=0.7).strip()


def keyword_flags(text):
    normalized = re.sub(r"[-‐‑]", " ", text.lower())  # "crime-ridden" matches "crime ridden"
    flags = [f"{term} ({kind})" for term, kind in STEREOTYPE_TERMS.items() if re.search(rf"\b{term}\b", normalized)]
    if IDENTITY_OPENER.search(text):
        flags.append("opens with race, ethnicity, age, income or disability as the defining trait")
    return flags


def review_profile(text, llm=None):
    """Keyword heuristics always; an LLM review too when llm is given. flagged is True if either flags it."""
    flags = keyword_flags(text)
    llm_review = None
    if llm is not None:
        raw = llm.chat([{"role": "system", "content": REVIEW_SYSTEM}, {"role": "user", "content": text}],
                       temperature=0, json_mode=True)
        try:
            llm_review = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            llm_review = {"error": "reviewer reply was not JSON", "raw": str(raw)[:200]}
    flagged = bool(flags) or bool(isinstance(llm_review, dict) and llm_review.get("stereotype") is True)
    return {"flagged": flagged, "keyword_flags": flags, "llm_review": llm_review}
