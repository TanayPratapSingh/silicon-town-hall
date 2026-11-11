"""Meetings and memory: turn order, short term memory, comment fields and the simulated label,
long term memory replace and delete. Offline: FakeLLM, a fake retriever and a fake embedder.
"""

import re
import zlib

import numpy as np

from townhall.llm import FakeLLM
from townhall.town import meeting, pilot
from townhall.town.memory import Memory

COMMENT_FIELDS = {"resident_id", "neighborhood", "text", "cited_pages", "stance", "simulated"}


def residents(n=3):
    return [{"resident_id": f"r{i}", "neighborhood": f"Hood {i}", "profile": f"I am resident {i}.",
             "commute": "Bus", "tenure": "Rented", "disability": "Without a disability"} for i in range(n)]


def responder(speak_stance="oppose", cited=(1, 99, "x", True)):
    def reply(messages):
        system = messages[0]["content"]
        if system == meeting.AGENDA_SYSTEM:
            return {"agenda": ["Traffic", "Noise"]}
        if system == meeting.SPEAK_SYSTEM:
            return {"text": "I spoke.", "stance": speak_stance, "cited": list(cited)}
        if system == meeting.SUMMARY_SYSTEM:
            return "summary"
        return {"text": "My comment.", "stance": "mixed", "reasons": ["noise"], "cited": [2]}
    return reply


class RecordingRetriever:
    def __init__(self):
        self.calls = []

    def __call__(self, query, cited_pages):
        self.calls.append((query, list(cited_pages)))
        return pilot.fake_retriever(query, cited_pages)


def fake_embed(texts):
    out = np.zeros((len(texts), 64))
    for i, text in enumerate(texts):
        for word in re.findall(r"[a-z]+", text.lower()):
            out[i, zlib.crc32(word.encode()) % 64] += 1
    return out


def test_facilitator_calls_on_residents_in_turn():
    state = meeting.run_meeting("policy", residents(3), RecordingRetriever(), FakeLLM(responder=responder()), max_turns=5)
    assert [t["resident_id"] for t in state["transcript"]] == ["r0", "r1", "r2", "r0", "r1"]
    assert state["agenda"] == ["Traffic", "Noise"]
    assert [t["agenda_item"] for t in state["transcript"]] == ["Traffic", "Traffic", "Traffic", "Noise", "Noise"]


def test_default_turns_is_one_per_resident():
    state = meeting.run_meeting("policy", residents(4), RecordingRetriever(), FakeLLM(responder=responder()))
    assert len(state["transcript"]) == 4


def test_short_term_memory_and_cited_page_boost():
    retriever = RecordingRetriever()
    state = meeting.run_meeting("policy", residents(3), retriever, FakeLLM(responder=responder()), max_turns=6)
    assert len(state["recent"]) == meeting.RECENT_TURNS
    assert state["summary"] == "summary"
    assert retriever.calls[0][1] == []                 # nothing cited before the first turn
    assert retriever.calls[1][1] == [tuple(state["transcript"][0]["cited_pages"][0])]
    assert "I spoke." in retriever.calls[1][0]          # the query carries what the speaker just heard


def test_citations_outside_the_excerpts_are_dropped():
    state = meeting.run_meeting("policy", residents(2), RecordingRetriever(), FakeLLM(responder=responder()))
    fake_pages = [[p["file"], p["page"]] for p in pilot.FAKE_PAGES]
    for turn in state["transcript"]:
        assert len(turn["cited_pages"]) == 1 and turn["cited_pages"][0] in fake_pages  # 99, "x" and True dropped


def test_stance_can_change_and_bad_replies_are_counted():
    state = meeting.run_meeting("policy", residents(2), RecordingRetriever(), FakeLLM(responder=responder("oppose")))
    assert state["stance_changes"] == 2 and state["parse_failures"] == 0
    state = meeting.run_meeting("policy", residents(2), RecordingRetriever(), FakeLLM(reply="not json"))
    assert state["stance_changes"] == 0
    assert state["parse_failures"] == 1 + 2 + 2          # agenda, two turns, two comments
    assert {c["stance"] for c in state["comments"]} == {"undecided"}


def test_one_comment_each_labeled_simulated():
    state = meeting.run_meeting("policy", residents(3), RecordingRetriever(), FakeLLM(responder=responder()))
    comments = state["comments"]
    assert [c["resident_id"] for c in comments] == ["r0", "r1", "r2"]
    for c in comments:
        assert set(c) == COMMENT_FIELDS
        assert c["simulated"] is True
        assert c["stance"] == "mixed" and c["text"] == "My comment." and len(c["cited_pages"]) == 1
    assert all(t["simulated"] is True for t in state["transcript"])


def test_pilot_fakes_cover_every_prompt():
    state = meeting.run_meeting(pilot.POLICY, residents(3), pilot.fake_retriever, FakeLLM(responder=pilot.fake_responder))
    assert len(state["comments"]) == 3 and state["parse_failures"] == 0


def test_retriever_adapter_shape():
    class Hit:
        def __init__(self, page):
            self.doc_id, self.file, self.page, self.text = "doc", "doc.pdf", page, "text"

    class Searcher:
        def search(self, query, k=10, mode="both", cited_pages=None, llm=None):
            self.args = (query, k, mode, cited_pages)
            return [Hit(3), {"doc_id": "doc", "file": "doc.pdf", "page": 4, "text": "t"}]

    searcher = Searcher()
    hits = meeting.retriever_from(searcher, k=5)("q", [("doc", 1)])
    assert searcher.args == ("q", 5, "both", [("doc", 1)])
    assert [meeting.page_key(h) for h in hits] == [["doc", 3], ["doc", 4]]


def test_memory_newer_round_replaces_older(tmp_path):
    memory = Memory(tmp_path / "m.sqlite", embed=fake_embed)
    assert memory.remember("r0", "I-81", "oppose", ["noise"], "Too loud.", round=1)
    assert memory.remember("r0", "I-81", "support", ["jobs"], "Changed my mind.", round=2)
    assert not memory.remember("r0", "I-81", "oppose", ["stale"], "Old opinion.", round=1)
    memory.remember("r0", "bus routes", "mixed", ["late buses"], "Buses run late.", round=1)
    assert memory.count("r0") == 2
    recalled = memory.recall("r0", "I-81 viaduct opinion", k=3)
    i81 = [m for m in recalled if m["topic"] == "I-81"]
    assert len(i81) == 1 and i81[0]["stance"] == "support" and i81[0]["round"] == 2
    assert memory.recall("r0", "late buses", k=1)[0]["topic"] == "bus routes"
    assert memory.recall("nobody", "anything") == []


def test_memory_delete_resident_and_all(tmp_path):
    memory = Memory(tmp_path / "m.sqlite", embed=fake_embed)
    for rid in ("r0", "r1"):
        memory.remember(rid, "I-81", "oppose", [], "comment", round=1)
    assert memory.delete_resident("r0") == 1
    assert memory.count("r0") == 0 and memory.count("r1") == 1
    assert memory.delete_all() == 1 and memory.count() == 0


def test_meeting_saves_and_recalls_memory(tmp_path):
    memory = Memory(tmp_path / "m.sqlite", embed=fake_embed)
    llm = FakeLLM(responder=responder())
    meeting.run_meeting("policy", residents(2), RecordingRetriever(), llm, memory=memory, round=1, topic="I-81")
    assert memory.count() == 2 and memory.recall("r0", "I-81")[0]["stance"] == "mixed"
    llm.calls.clear()
    state = meeting.run_meeting("policy", residents(2), RecordingRetriever(), llm, memory=memory, round=2, topic="I-81")
    speak_prompt = next(m[1]["content"] for m in llm.calls if m[0]["content"] == meeting.SPEAK_SYSTEM)
    assert "Your stance so far: mixed" in speak_prompt and "round 1, I-81: mixed" in speak_prompt
    assert state["memories"]["r0"][0]["round"] == 1
    assert memory.recall("r0", "I-81")[0]["round"] == 2
