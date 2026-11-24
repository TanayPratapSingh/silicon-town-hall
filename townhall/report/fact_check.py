"""Do simulated turns and comments contradict the resident's own Census record? (issue #21)

Phrase matching on four facts the record states plainly: commute, own or rent, age and disability. It only
catches first person claims it has a pattern for, so its rate is a lower bound, not a measured error rate.
No model calls, so it costs nothing and can run on every saved transcript.
"""

import re
from collections import Counter

from townhall.town import census

COMMUTE_CLAIM = re.compile(r"\b(my (daily |morning |work )?commute|commute (to|from) work|commuting to work|"
                           r"(drive|bike|walk|ride the bus|take the bus) to work|on my way to work)\b", re.I)
RENT_CLAIM = re.compile(r"\b(I rent\b|I'm renting|I am renting|as a renter|my landlord|my rent\b)", re.I)
OWN_CLAIM = re.compile(r"\b(I own my (home|house)|as an? home ?owner|I'm a home ?owner|I am a home ?owner|"
                       r"my mortgage)\b", re.I)
DISABILITY_CLAIM = re.compile(r"\b(my disability|I have a disability|as (a person|someone) with a disability|"
                              r"as a disabled|I'm disabled|I am disabled|my wheelchair)\b", re.I)
AGE_CLAIM = re.compile(r"\b(\d{2})[- ]years?[- ]old\b|\b(?:I am|I'm) (\d{2})(?=[,.;]| and | years\b)", re.I)
KINDS = ("commute", "tenure", "age", "disability")


def check(text, resident):
    """Kinds of contradiction between one text and its resident's record, in KINDS order."""
    found = set()
    if resident.get("commute") == census.NO_COMMUTE and COMMUTE_CLAIM.search(text):
        found.add("commute")
    tenure = resident.get("tenure", "")
    if (tenure.startswith("Owned") and RENT_CLAIM.search(text)) or (
            tenure in ("Rented", census.NO_TENURE) and OWN_CLAIM.search(text)):
        found.add("tenure")
    ages = [int(a or b) for a, b in AGE_CLAIM.findall(text)]
    if "age" in resident and any(abs(a - int(resident["age"])) > 1 for a in ages):
        found.add("age")
    if resident.get("disability") == "Without a disability" and DISABILITY_CLAIM.search(text):
        found.add("disability")
    return [k for k in KINDS if k in found]


def check_texts(items, residents):
    """items: dicts with resident_id and text (turns or comments). Counts only, no text."""
    by_id = {r["resident_id"]: r for r in residents}
    kinds = [check(i["text"], by_id[i["resident_id"]]) for i in items if i["resident_id"] in by_id]
    return {"texts_checked": len(kinds), "texts_with_a_contradiction": sum(bool(k) for k in kinds),
            "by_kind": {k: n for k, n in Counter(x for ks in kinds for x in ks).items()},
            "method": "phrase matching against the Census record; a lower bound"}
