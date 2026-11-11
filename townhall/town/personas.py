"""Persona builder agent: a Census record becomes a short first person profile, then a stereotype review.

The review has two parts: keyword heuristics that always run, and an optional LLM reviewer. The
heuristics only catch wording on the list below; a profile that passes them can still lean on a
stereotype, which is why the LLM hook exists and why profiles still need a human read.
"""

import json
import re

from townhall.town import census

PROFILE_FIELDS = ["age", "sex", "race", "hispanic_origin", "income", "commute", "tenure", "disability", "neighborhood"]
HAS_DISABILITY = "With a disability"
NO_COMMUTE_FACT = "did not commute to work last week (the record does not say whether the person has a job)"
GROUP_QUARTERS_FACT = "lives in group quarters (type not recorded)"

PROFILE_SYSTEM = (
    "You write a short first person profile of a simulated resident for a public planning exercise. "
    "Use only the facts given; leave out anything they do not state. Write four to six plain sentences. "
    "Do not start with race or ethnicity, but one later sentence must state the resident's race and Hispanic "
    "origin plainly, in the words given, as part of who they are and never as the reason for anything else. "
    "State income as yearly income in dollars. Mention disability only if a disability fact is given, and then "
    "only in one neutral clause, without saying what the person can or cannot do or whether they need help. "
    "Never mention not having a disability. The commute fact does not say whether the person has a job, so "
    "never state employment status: do not say employed, unemployed, retired, working or not working, or "
    "having or not having a job. The type of group quarters is not recorded, so never name one, such as a "
    "nursing home, dorm or prison. Do not invent health, family, a job or employer, religion, personality, "
    "culture, opinions, feelings, or whether the money is enough. Do not describe low income or disability as "
    "tragedy or inspiration."
)

REVIEW_SYSTEM = (
    "You review a simulated resident profile, written from the Census facts given, for stereotypes and "
    "invented facts. Flag as stereotype: traits attributed to the person because of race, ethnicity, sex, age, "
    "income or disability; race or ethnicity used to explain views or circumstances, or as the opening or "
    "defining trait; deficit, dependence or inspiration framing of disability or poverty; any mention of not "
    "having a disability. Flag as invented: any claim the facts do not state, such as health, family, job, "
    "religion, opinions, feelings or whether money is enough, any employment status (the facts never give one), "
    "any named type of group quarters such as a nursing home, dorm or prison, and income given as anything but "
    "yearly. Reply "
    'as JSON: {"stereotype": true or false, "invented_facts": true or false, "reasons": ["short reason", ...]}.'
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
# Patterns from the first real profile run (runs/town_profiles_real.json, Nov 11, 2026), checked per sentence.
SENTENCE_FLAGS = {
    "mentions not having a disability": re.compile(r"(\b(no|without|not have|free of)|n't have) (a |any )?disabilit"),
    "dependence framing of disability": re.compile(
        r"disabilit[^.]*\b(independent\w*|on my own|by myself|without (special )?(help|assistance|accommodations?|"
        r"difficulty)|need help|rely on|depend\w*|manage)\b|\b(independent\w*|on my own|need help|rely on|depend\w*)\b"
        r"[^.]*disabilit"),
    "invented health claim": re.compile(
        r"\b(good|great|excellent|fine|poor|bad|declining|failing) health\b|\bhealthy\b|\bstay active\b"),
    "states employment status": re.compile(
        r"\b(un)?employ(ed|ment)\b|\b(do not|don't|does not|doesn't) have a job\b|\bno job\b|\bjobless\b|"
        r"\bnot (currently )?working\b|\bretired\b|\bretirement\b|\bout of work\b|\bwork as an?\b"),
    "names a type of group quarters": re.compile(
        r"\b(nursing home|dorm(itory|itories|s)?|residence hall|prison|jail|correctional|shelter|barracks|"
        r"group home|assisted living)\b"),
    "income given as monthly, weekly or hourly": re.compile(
        r"\b(monthly|weekly|hourly) (income|earnings|pay|salary)\b|"
        r"\b(income|earn\w*|salary)\b[^.]{0,40}\b(per|a|an|each) (month|week|hour)\b"),
}
IDENTITY_OPENER = re.compile(
    r"^\W*as an? (black|white|asian|hispanic|latina|latino|latinx|african american|native|indigenous|"
    r"puerto rican|mexican|dominican|disabled|immigrant|old|elderly|young|poor|low income)\b",
    re.IGNORECASE,
)


def plain_race(label):
    """'White alone' -> 'White'; the long American Indian label -> 'American Indian or Alaska Native'."""
    return "American Indian or Alaska Native" if label.startswith("American Indian and") else label.removesuffix(" alone")


def plain_hispanic(label):
    if label == "Not Spanish/Hispanic/Latino":
        return "not Hispanic or Latino"
    return "Hispanic or Latino" if label.startswith("All Other") else f"Hispanic or Latino ({label})"


def facts(resident):
    """The fields the profile writer may use, in plain words. Serial numbers and weights stay out."""
    out = {key: resident.get(key) for key in PROFILE_FIELDS}
    out["race"], out["hispanic_origin"] = plain_race(out["race"]), plain_hispanic(out["hispanic_origin"])
    out["annual_income_dollars"] = out.pop("income")  # a bare "income" was read as monthly in the first real run
    # the Census N/A labels led profiles to say "I do not have a job" and "such as a nursing home" in the second run
    if out["commute"] == census.NO_COMMUTE:
        out["commute"] = NO_COMMUTE_FACT
    if out["tenure"] == census.NO_TENURE:
        out["tenure"] = GROUP_QUARTERS_FACT
    if out.pop("disability") == HAS_DISABILITY:  # no disability is left out, so the writer cannot frame it
        out["disability"] = "has a disability (type not recorded)"
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
    sentences = re.split(r"(?<=[.!?])\s+", normalized.replace("\u2019", "'"))
    flags += [name for name, pattern in SENTENCE_FLAGS.items() if any(pattern.search(s) for s in sentences)]
    return flags


def review_profile(text, llm=None, resident=None):
    """Keyword heuristics always; an LLM review too when llm is given, against the resident's facts when given.
    flagged is True if the keywords or the reviewer flag a stereotype or an invented fact."""
    flags = keyword_flags(text)
    llm_review = None
    if llm is not None:
        given = f"Facts (JSON): {json.dumps(facts(resident))}\n" if resident else ""
        raw = llm.chat([{"role": "system", "content": REVIEW_SYSTEM},
                        {"role": "user", "content": f"{given}Profile: {text}"}], temperature=0, json_mode=True)
        try:
            llm_review = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            llm_review = {"error": "reviewer reply was not JSON", "raw": str(raw)[:200]}
    flagged = bool(flags) or (isinstance(llm_review, dict) and (llm_review.get("stereotype") is True
                                                                 or llm_review.get("invented_facts") is True))
    return {"flagged": flagged, "keyword_flags": flags, "llm_review": llm_review}
