"""One simulated neighborhood meeting as a LangGraph loop.

agenda -> call_on -> speak -> summarize -> (call_on again, or comments at the end)

The facilitator sets the agenda from the policy and calls on residents in turn, round robin, for
max_turns turns. Each resident builds a query from its profile and the last thing it heard, calls the
injected retriever, speaks, and may change its stance. Short term memory lives in the graph state:
the last few turns, a running summary, and the pages already cited (passed back to the retriever as a
boost). At the end every resident writes one comment, labeled simulated. With a Memory store, each
resident recalls its top 3 past opinions before the meeting and saves its new one after.

retriever(query, cited_pages) -> list of hits, each a dict with file, page, text (doc_id optional).
cited_pages is a list of (doc_id or file, page) pairs. Pages are cited as [doc_id or file, page].
"""

import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from townhall.town import personas

STANCES = ("support", "oppose", "mixed", "undecided")
RECENT_TURNS = 4
EXCERPT_CHARS = 800

AGENDA_SYSTEM = (
    "You are the facilitator of a simulated neighborhood meeting about a draft policy. Set an agenda of "
    'three to five short items residents should discuss. Reply as JSON: {"agenda": ["item", ...]}.'
)
SPEAK_SYSTEM = (
    "You are a simulated resident speaking at a neighborhood meeting about a draft policy. Speak in first "
    "person as the resident in the profile, in two to four sentences, about the current agenda item and "
    "how it touches your own life. Ground what you say in the numbered policy excerpts and cite the ones "
    "you used. You may respond to what others said and you may change your stance if persuaded. Reply as "
    'JSON: {"text": "...", "stance": "support" or "oppose" or "mixed" or "undecided", "cited": [excerpt numbers]}.'
)
SUMMARY_SYSTEM = (
    "You keep the running summary of a simulated neighborhood meeting. Fold the new turn into the summary "
    "in at most five sentences, keeping who raised which concern. Reply with the summary only."
)
COMMENT_SYSTEM = (
    "You are a simulated resident writing one public comment on a draft policy after a neighborhood "
    "meeting. Write three to five sentences in first person, grounded in the numbered policy excerpts, and "
    'cite the ones you used. Reply as JSON: {"text": "...", "stance": "support" or "oppose" or "mixed" or '
    '"undecided", "reasons": ["short reason", ...], "cited": [excerpt numbers]}.'
)


class MeetingState(TypedDict):
    policy: str
    residents: list
    memories: dict        # resident_id -> recalled long term memories
    agenda: list
    max_turns: int
    turn: int
    speaker: int
    transcript: list
    recent: list          # short term memory: the last RECENT_TURNS turns
    summary: str          # short term memory: running summary
    cited_pages: list     # short term memory: pages cited so far in this meeting
    resident_pages: dict  # resident_id -> pages that resident cited while speaking
    stances: dict
    stance_changes: int
    parse_failures: int
    comments: list
    reasons: dict


def retriever_from(searcher, k=5, mode="both", llm=None):
    """Adapt townhall.retrieval.Retriever.search(query, k, mode, cited_pages, llm) to retriever(query, cited_pages)."""
    def retrieve(query, cited_pages):
        hits = searcher.search(query, k=k, mode=mode, cited_pages=cited_pages, llm=llm)
        fields = ("doc_id", "file", "page", "text")
        return [{f: (h.get(f) if isinstance(h, dict) else getattr(h, f, None)) for f in fields} for h in hits]
    return retrieve


def build_query(resident, item, heard):
    """Retrieval query from the agenda item, the resident's own circumstances and the last turn it heard."""
    # shortcut: a template, not an LLM written query; switch to one if retrieval misses what residents care about
    own = "; ".join(str(resident.get(key, "")) for key in ("neighborhood", "commute", "tenure", "disability"))
    return f"{item}. {own}. {heard[:300]}".strip()


def page_key(hit):
    return [hit.get("doc_id") or hit["file"], int(hit["page"])]


def _union(pages, more):
    """Pages in first seen order, without repeats."""
    out = [list(p) for p in pages]
    for page in more:
        if list(page) not in out:
            out.append(list(page))
    return out


def _excerpts(hits):
    return "\n".join(f"[{i}] {h['file']} page {h['page']}: {(h.get('text') or '')[:EXCERPT_CHARS]}"
                     for i, h in enumerate(hits, 1)) or "(no excerpts found)"


def _parse(raw, hits, stance):
    """Reply JSON to (text, stance, cited pages, reasons, ok). Citations outside the excerpts are dropped."""
    try:
        out = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        out = None
    if not isinstance(out, dict):
        return str(raw).strip(), stance, [], [], False
    cited = [page_key(hits[i - 1]) for i in out.get("cited") or []
             if isinstance(i, int) and not isinstance(i, bool) and 1 <= i <= len(hits)]
    new_stance = out.get("stance") if out.get("stance") in STANCES else stance
    reasons = [str(r) for r in out.get("reasons") or []]
    return str(out.get("text", "")).strip(), new_stance, _union([], cited), reasons, True


def _facts(resident):
    """The resident's Census facts as a prompt line (turns stated commutes the record does not have)."""
    if "race" not in resident:  # test residents carry only an id and a profile
        return ""
    return ("Census facts about you (true; never say anything that contradicts them, such as a commute, job, home "
            f"or age the record does not have): {json.dumps(personas.facts(resident))}\n")


def _remembered(memories):
    return "\n".join(f"- round {m['round']}, {m['topic']}: {m['stance']}. {m['comment']}" for m in memories) or "nothing yet"


def build_graph(llm, retriever):
    def agenda(state):
        raw = llm.chat([{"role": "system", "content": AGENDA_SYSTEM},
                        {"role": "user", "content": f"Policy: {state['policy']}"}], temperature=0.3, json_mode=True)
        try:
            items = [str(i) for i in json.loads(raw).get("agenda", []) if str(i).strip()][:5]
        except (json.JSONDecodeError, TypeError, AttributeError):
            items = []
        return {"agenda": items or [state["policy"]], "parse_failures": state["parse_failures"] + (not items)}

    def call_on(state):
        return {"speaker": state["turn"] % len(state["residents"])}

    def speak(state):
        resident = state["residents"][state["speaker"]]
        rid = resident["resident_id"]
        item = state["agenda"][state["turn"] * len(state["agenda"]) // state["max_turns"]]
        heard = state["recent"][-1]["text"] if state["recent"] else ""
        hits = retriever(build_query(resident, item, heard), [tuple(p) for p in state["cited_pages"]])
        last = "\n".join(f"{t['resident_id']} ({t['neighborhood']}): {t['text']}" for t in state["recent"])
        prompt = (f"Your profile: {resident['profile']}\n{_facts(resident)}Policy: {state['policy']}\nAgenda item: {item}\n"
                  f"Your stance so far: {state['stances'][rid]}\n"
                  f"What you said in earlier rounds:\n{_remembered(state['memories'].get(rid, []))}\n"
                  f"Meeting summary so far: {state['summary'] or 'the meeting just started'}\n"
                  f"Last turns:\n{last or '(you speak first)'}\nPolicy excerpts:\n{_excerpts(hits)}")
        raw = llm.chat([{"role": "system", "content": SPEAK_SYSTEM}, {"role": "user", "content": prompt}], json_mode=True)
        text, stance, pages, _, ok = _parse(raw, hits, state["stances"][rid])
        turn = {"turn": state["turn"], "resident_id": rid, "neighborhood": resident.get("neighborhood"),
                "agenda_item": item, "text": text, "stance": stance, "cited_pages": pages, "simulated": True}
        return {
            "transcript": state["transcript"] + [turn],
            "recent": (state["recent"] + [turn])[-RECENT_TURNS:],
            "cited_pages": _union(state["cited_pages"], pages),
            "resident_pages": {**state["resident_pages"], rid: _union(state["resident_pages"].get(rid, []), pages)},
            "stances": {**state["stances"], rid: stance},
            "stance_changes": state["stance_changes"] + (stance != state["stances"][rid]),
            "parse_failures": state["parse_failures"] + (not ok),
            "turn": state["turn"] + 1,
        }

    def summarize(state):
        turn = state["transcript"][-1]
        prompt = (f"Summary so far: {state['summary'] or 'none'}\n"
                  f"New turn by {turn['resident_id']} ({turn['neighborhood']}): {turn['text']}")
        summary = llm.chat([{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": prompt}],
                           temperature=0.2)
        return {"summary": summary.strip()}

    def comments(state):
        out, reasons, failures = [], {}, 0
        for resident in state["residents"]:
            rid = resident["resident_id"]
            own_pages = state["resident_pages"].get(rid, [])
            hits = retriever(build_query(resident, "overall comment on the policy", state["summary"]),
                             [tuple(p) for p in own_pages])
            prompt = (f"Your profile: {resident['profile']}\n{_facts(resident)}Policy: {state['policy']}\n"
                      f"Your stance after the meeting: {state['stances'][rid]}\n"
                      f"Meeting summary: {state['summary']}\nPolicy excerpts:\n{_excerpts(hits)}")
            raw = llm.chat([{"role": "system", "content": COMMENT_SYSTEM}, {"role": "user", "content": prompt}],
                           json_mode=True)
            text, stance, pages, reasons[rid], ok = _parse(raw, hits, state["stances"][rid])
            failures += not ok
            out.append({"resident_id": rid, "neighborhood": resident.get("neighborhood"), "text": text,
                        "cited_pages": pages, "stance": stance, "simulated": True})
        return {"comments": out, "reasons": reasons, "parse_failures": state["parse_failures"] + failures}

    graph = StateGraph(MeetingState)
    for name, node in [("agenda", agenda), ("call_on", call_on), ("speak", speak),
                       ("summarize", summarize), ("comments", comments)]:
        graph.add_node(name, node)
    graph.add_edge(START, "agenda")
    graph.add_edge("agenda", "call_on")
    graph.add_edge("call_on", "speak")
    graph.add_edge("speak", "summarize")
    graph.add_conditional_edges("summarize", lambda s: "call_on" if s["turn"] < s["max_turns"] else "comments",
                                ["call_on", "comments"])
    graph.add_edge("comments", END)
    return graph.compile()


def run_meeting(policy, residents, retriever, llm, max_turns=None, memory=None, round=1, topic=None):
    """Run one meeting and return its final state. Every resident needs resident_id and profile."""
    if not residents:
        raise ValueError("a meeting needs at least one resident")
    topic = topic or policy
    max_turns = max_turns or len(residents)
    memories = {r["resident_id"]: memory.recall(r["resident_id"], topic, k=3) for r in residents} if memory else {}
    # a resident starts from its saved stance on this topic, so an opinion carries into the next round
    stances = {r["resident_id"]: next((m["stance"] for m in memories.get(r["resident_id"], []) if m["topic"] == topic),
                                      "undecided") for r in residents}
    state = {"policy": policy, "residents": residents, "memories": memories, "agenda": [], "max_turns": max_turns,
             "turn": 0, "speaker": 0, "transcript": [], "recent": [], "summary": "", "cited_pages": [],
             "resident_pages": {}, "stances": stances, "stance_changes": 0, "parse_failures": 0,
             "comments": [], "reasons": {}}
    final = build_graph(llm, retriever).invoke(state, config={"recursion_limit": 3 * max_turns + 10})
    if memory:
        for c in final["comments"]:
            memory.remember(c["resident_id"], topic, c["stance"], final["reasons"].get(c["resident_id"], []),
                            c["text"], round)
    return final
