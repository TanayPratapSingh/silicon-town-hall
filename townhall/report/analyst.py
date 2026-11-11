"""Analyst agent: tag simulated comments with the answer key's codebook and tagger, then build the report.

The report counts topics by neighborhood and by demographic group, shows sample comments (each labeled
simulated, with its page citations), and lists who to reach in person: groups and neighborhoods with no
simulated resident, fewer than half the residents their population share implies, only one simulated
voice, or simulated comments that raised no codebook topic.
"""

from collections import Counter

from townhall.ingest.tagger import tag

DIMENSIONS = ("race_ethnicity", "age_band", "tenure", "income_band", "disability", "commute")
UNDER_RATIO = 0.5  # a group is underrepresented below half the residents its population share implies
SIMULATED_LABEL = "SIMULATED: written by an AI resident, not a real person and not public input"
AGE_BANDS = ((30, "18 to 29"), (45, "30 to 44"), (65, "45 to 64"), (200, "65 and over"))
INCOME_BANDS = ((25_000, "under $25,000"), (50_000, "$25,000 to $49,999"), (100_000, "$50,000 to $99,999"),
                (float("inf"), "$100,000 and over"))


def _band(value, bands):
    return next(label for upper, label in bands if value < upper)


def groups(person):
    """The resident's group on each demographic dimension, plus its neighborhood when it has one."""
    out = {"race_ethnicity": person["race_ethnicity"], "age_band": _band(person["age"], AGE_BANDS),
           "tenure": person["tenure"], "income_band": _band(person["income"], INCOME_BANDS),
           "disability": person["disability"], "commute": person["commute"]}
    return {**out, "neighborhood": person["neighborhood"]} if "neighborhood" in person else out


def tag_comments(comments, codebook, llm):
    """Copies of the comments with the codes the answer key's tagger gives them (one LLM call each)."""
    return [{**c, "codes": tag(c["text"], codebook, llm)} for c in comments]


def population_shares(state="36", puma="00701"):
    """Weighted share of adults in each group (ACS PUMS) and of tract population in each neighborhood."""
    from townhall.town import census, homes  # loads Census files or calls the API; tests pass shares in

    people, source = census.load_people(state, puma)
    adults = [census.decode(row) for row in census.adults(people).to_dict("records")]
    total = sum(a["person_weight"] for a in adults)
    shares = {dim: Counter() for dim in DIMENSIONS}
    for a in adults:
        for dim, value in groups(a).items():
            shares[dim][value] += a["person_weight"] / total
    tracts, tract_source = homes.load_tracts(state, puma)
    hoods = tracts.groupby("neighborhood").population.sum()
    shares["neighborhood"] = (hoods / hoods.sum()).to_dict()
    return {dim: {k: round(float(v), 4) for k, v in s.items()} for dim, s in shares.items()}, f"{source}; {tract_source}"


def _stats(tagged, residents):
    """dimension -> group -> residents, comments, comments with no topic, topic counts."""
    by_id = {r["resident_id"]: r for r in residents}
    out = {}
    for r in residents:
        for dim, value in groups(r).items():
            out.setdefault(dim, {}).setdefault(value, {"residents": 0, "comments": 0, "no_topic_comments": 0,
                                                       "topics": Counter()})["residents"] += 1
    for c in tagged:
        for dim, value in groups(by_id[c["resident_id"]]).items():
            cell = out[dim][value]
            cell["comments"] += 1
            cell["no_topic_comments"] += not c["codes"]
            cell["topics"].update(c["codes"])
    return out


def who_to_reach(stats, n_residents, population=None):
    """Groups and neighborhoods to reach in person, biggest shortfall against population first."""
    out = []
    for dim in set(stats) | set(population or {}):
        cells, shares = stats.get(dim, {}), (population or {}).get(dim, {})
        for group in set(cells) | set(shares):
            cell, share = cells.get(group, {}), shares.get(group)
            count = cell.get("residents", 0)
            expected = n_residents * share if share is not None else None
            reasons = []
            if count == 0 and share:
                reasons.append(f"no simulated resident, though the group is {share:.1%} of the population")
            elif expected is not None and count < UNDER_RATIO * expected:
                reasons.append(f"underrepresented: {count} simulated residents where its population share implies "
                               f"{expected:.1f}")
            if count == 1:
                reasons.append("only one simulated resident speaks for this group")
            if count and cell["comments"] == cell["no_topic_comments"]:
                reasons.append("its simulated comments raised no codebook topic")
            if reasons:
                out.append({"dimension": dim, "group": group, "simulated_residents": count,
                            "expected_residents": None if expected is None else round(expected, 2),
                            "population_share": share, "reasons": reasons})
    return sorted(out, key=lambda e: (-((e["expected_residents"] or 0) - e["simulated_residents"]),
                                      e["dimension"], str(e["group"])))


def samples(tagged, per_topic=1):
    """Up to per_topic comments for each topic, only comments that cite pages, each labeled simulated."""
    out, used = [], Counter()
    for c in tagged:
        for code in c["codes"]:
            if c["cited_pages"] and used[code] < per_topic:
                used[code] += 1
                out.append({"label": SIMULATED_LABEL, "simulated": True, "topic": code, "resident_id": c["resident_id"],
                            "neighborhood": c.get("neighborhood"), "stance": c.get("stance"), "text": c["text"],
                            "cited_pages": c["cited_pages"]})
    return out


def build_report(tagged, residents, codebook, population=None, population_source=None):
    """Report dict for the app. tagged: comments with codes; residents: their Census records."""
    if not all(c.get("simulated") is True for c in tagged):
        raise ValueError("the analyst only reports on comments labeled simulated")
    stats = _stats(tagged, residents)
    plain = {dim: {g: {**cell, "topics": dict(cell["topics"].most_common())} for g, cell in cells.items()}
             for dim, cells in stats.items()}
    return {
        "label": SIMULATED_LABEL,
        "topic_names": {t["code"]: t["topic"] for t in codebook},
        "residents": len(residents),
        "comments": len(tagged),
        "comments_with_no_topic": sum(not c["codes"] for c in tagged),
        "topics": dict(Counter(code for c in tagged for code in c["codes"]).most_common()),
        "by_neighborhood": plain.pop("neighborhood", {}),
        "by_group": plain,
        "samples": samples(tagged),
        "who_to_reach": who_to_reach(stats, len(residents), population),
        "population_source": population_source if population else "population shares not available",
    }
