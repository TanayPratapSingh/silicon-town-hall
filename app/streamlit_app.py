"""Silicon Town Hall app (issue #9). Run: streamlit run app/streamlit_app.py

Behind a password; every page carries the simulated banner. The sidebar switch "Use the free test model"
starts ON and then FakeLLM writes every profile, turn and comment, so nothing costs money.
"""

import hmac
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `app` and `townhall` import

import pandas as pd
import streamlit as st

from app import core
from townhall import config

BANNER = "Simulated residents. Not public comment."
PAGES = ["Community", "Policy", "Run meetings", "Report", "Saved scenarios"]
PRESET_CHOICE, UPLOAD_CHOICE = "I-81 impact statement (preset, already indexed)", "Upload a PDF"


def expected_password():
    """st.secrets on Streamlit Cloud, else APP_PASSWORD from .env."""
    if st.secrets.load_if_toml_exists() and "APP_PASSWORD" in st.secrets:
        return str(st.secrets["APP_PASSWORD"])
    return config.APP_PASSWORD


def gate():
    """Stops the script until the right password is typed; nothing else renders before that."""
    if st.session_state.get("authed"):
        return
    st.title("Silicon Town Hall")
    expected = expected_password()
    if not expected:
        st.error("APP_PASSWORD is not set, so the app stays locked.")
        st.stop()
    typed = st.text_input("Password", type="password")
    if typed and hmac.compare_digest(typed.encode(), expected.encode()):
        st.session_state.authed = True
        st.rerun()
    if typed:
        st.error("Wrong password.")
    st.stop()


def pick_scenario(label, saved, index=0, key=None):
    """Selectbox over saved scenarios by id; returns the scenario."""
    by_id = {s["id"]: s for s in saved}
    sid = st.selectbox(label, list(by_id), index=index, key=key,
                       format_func=lambda i: f"{by_id[i]['name']} ({by_id[i]['date'][:16].replace('T', ' ')} UTC)")
    return by_id[sid]


def community_page(llm):
    st.header("1. Choose a community")
    place = st.selectbox("Community", list(core.PRESETS))
    n = st.number_input("Number of residents", min_value=1, max_value=300, value=20)
    seed = st.number_input("Seed", min_value=0, value=0)
    if st.button("Sample residents", key="sample"):
        state, puma = core.PRESETS[place]
        with st.spinner("Sampling Census records and writing profiles"):
            residents, meta = core.sample_community(state, puma, int(n), int(seed), llm)
        st.session_state.residents = residents
        st.session_state.community = {"community": place, "state": state, "puma": puma, "residents": int(n),
                                      "seed": int(seed), **meta, "profile_usage": core.usage(llm)}
    residents = st.session_state.get("residents")
    if not residents:
        st.info("Pick the size and seed, then sample. Residents are real Census records with simulated profiles.")
        return
    c = st.session_state.community
    st.caption(f"{c['community']}, {c['residents']} residents, seed {c['seed']}. Census: {c['census_source']}. "
               f"Profiles: {residents[0]['profile_written_by']}")
    st.dataframe(core.resident_table(residents), hide_index=True)


def policy_page():
    st.header("2. Policy")
    policy = st.session_state.get("policy", core.I81)
    choice = st.radio("Policy", [PRESET_CHOICE, UPLOAD_CHOICE], index=int(policy["kind"] == "upload"))
    if choice == PRESET_CHOICE:
        policy = st.session_state.policy = core.I81
    else:
        uploaded = st.file_uploader("Policy PDF", type="pdf")
        if uploaded and st.button("Index this PDF", key="index"):
            try:
                with st.spinner("Indexing every page as text and as a page image"):
                    policy = st.session_state.policy = core.index_upload(uploaded.name, uploaded.getvalue())
            except ValueError as e:
                st.error(str(e))
        if policy["kind"] != "upload":
            st.info("Upload a PDF and index it. It goes into its own collection, apart from the I-81 index.")
            return
    st.markdown(f"**Using:** {policy['name']}, {core.page_count(policy['db_dir'])} pages indexed.")
    if policy.get("pages_without_text"):
        st.warning(f"{policy['pages_without_text']} pages have no text layer and were not OCRed; residents can "
                   "find them only through page image search.")
    query = st.text_input("Search the policy")
    if query:
        for hit in core.get_retriever(policy["db_dir"]).search(query, k=5, mode="both", llm=None):
            thumb, text = st.columns([1, 4])
            if Path(hit.image_path).exists():
                thumb.image(hit.image_path, width=140)
            text.markdown(f"**{hit.file}, page {hit.page}**")
            text.write(hit.text[:600])


def run_page(llm, vision):
    st.header("3. Run the meetings")
    residents = st.session_state.get("residents")
    if not residents:
        st.info("Sample residents on the Community page first.")
        return
    policy = st.session_state.get("policy", core.I81)
    st.markdown(f"**Policy:** {policy['name']}")
    hoods = sorted({r["neighborhood"] for r in residents})
    chosen = st.multiselect("Neighborhoods (one meeting each)", hoods, default=hoods)
    rounds = st.number_input("Rounds", min_value=1, max_value=5, value=1)
    change = st.text_area("Try a change (plan editor stub): describe an edit to the policy. Residents hear it as "
                          "part of the policy; the indexed document is not edited.")
    name = st.text_input("Scenario name", value=f"{policy['topic']} test")
    attending = sum(r["neighborhood"] in chosen for r in residents)
    st.caption(f"{attending} of {len(residents)} residents attend {len(chosen)} meetings per round. "
               f"Text by: {core.written_by(llm)}")
    if st.button("Run meetings", key="run", disabled=not chosen):
        bar = st.progress(0.0, text="Starting")

        def progress(done, total, rnd, hood):
            bar.progress(done / total, text=f"Round {rnd}: {hood} (meeting {done + 1} of {total})")

        scenario = core.run_scenario(name, residents, policy, chosen, int(rounds), llm, vision=vision, change=change,
                                     community=st.session_state.get("community"), on_progress=progress)
        bar.progress(1.0, text="Done")
        st.session_state.current = scenario["id"]
        if scenario["status"] != "complete":
            st.error(f"The run {scenario['status']}. What finished was saved.")
        st.success(f"Saved scenario {scenario['name']}. Open the Report page.")
        if scenario["usage"]["meetings"]["real_calls"]:
            st.caption(f"Real model use: {scenario['usage']['meetings']}")


def render_tables(report):
    if report is None:
        st.error("This scenario has no report: building it failed after the meetings were saved.")
        return
    for table in report["tables"]:
        st.subheader(table["title"])
        st.dataframe(pd.DataFrame(table["rows"]), hide_index=True)


def render_report(scenario):
    report = scenario["report"]
    if scenario["status"] != "complete":
        st.error(f"This run {scenario['status']}")
    if report is None:
        return st.error("This scenario has no report: "
                        f"{scenario.get('report_error', 'building it failed after the meetings were saved')}.")
    st.caption(report["label"])
    render_pilot_scores(scenario)
    st.subheader("Who to reach in person")
    st.markdown("\n".join(f"- {item}" for item in report["reach_in_person"]) or "Nobody flagged.")
    render_tables(report)
    st.subheader("Sample comments (SIMULATED)")
    for c in report["sample_comments"]:
        st.markdown(f"**SIMULATED** comment by {c['resident_id']}, {c['neighborhood']}, stance {c['stance']}"
                    + (f", topic {c['topic']}" if c.get("topic") else ""))
        st.write(c["text"])
        st.caption(f"Cited: {c['cited']}")


def render_pilot_scores(scenario):
    """The recorded I-81 pilot scores, the only place where real comments exist to score against."""
    st.subheader("Pilot scores")
    scores = core.pilot_scores() if scenario["settings"]["policy_kind"] == "preset" else None
    if scores is None:
        return st.caption("Pilot scores exist only for the I-81 pilot, where the real comments are known.")
    st.caption(scores["label"])
    if scores["scores"]:
        pct = lambda v: "pending" if v is None else f"{v:.1%}"
        table, notes = st.columns([1, 1])
        table.dataframe(pd.DataFrame([{"condition": name.replace("_", " "), "real topics raised": pct(s.get("topic_coverage")),
                                       "topic mix gap": s.get("topic_mix_gap")} for name, s in scores["scores"].items()]),
                        hide_index=True)
        notes.caption("Real topics raised: share of the topics in the real comments that at least one simulated comment "
                      "raised. Topic mix gap: 0 is the same mix of topics, 1 is no topic in common. Conditions: the "
                      "meeting; one model with no personas; the same residents writing without the meeting.")


def report_page():
    st.header("4. Report")
    saved = core.list_scenarios()
    if not saved:
        st.info("Run the meetings first.")
        return
    ids = [s["id"] for s in saved]
    current = st.session_state.get("current")
    render_report(pick_scenario("Scenario", saved, index=ids.index(current) if current in ids else 0))


def saved_page():
    st.header("5. Saved scenarios")
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))
    saved = core.list_scenarios()
    if not saved:
        st.info("No saved scenarios.")
        return
    st.dataframe(pd.DataFrame([{"name": s["name"], "date": s["date"], "policy": s["settings"]["policy"],
                                "residents": len(s["residents"]), "meetings": len(s["meetings"]),
                                "rounds": s["settings"]["rounds"], "text by": s["settings"]["meeting_text_by"],
                                "status": s["status"]} for s in saved]), hide_index=True)
    s = pick_scenario("View a scenario", saved, key="view")
    sid = s["id"]
    with st.expander("Settings"):
        st.json(s["settings"])
    with st.expander("Meeting transcripts (SIMULATED)"):
        for m in s["meetings"]:
            st.markdown(f"**Round {m['round']}, {m['neighborhood']}.** Agenda: {'; '.join(m['agenda'])}")
            st.dataframe(pd.DataFrame([{"label": "SIMULATED", "resident_id": t["resident_id"], "stance": t["stance"],
                                        "text": t["text"], "cited": core.cite(t["cited_pages"])}
                                       for t in m["transcript"]]), hide_index=True)

    st.subheader("Saved memories (SIMULATED)")
    rows = core.memories(sid)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        rid = st.selectbox("Resident", sorted({r["resident_id"] for r in rows}))
        if st.button("Delete this resident's memories", key="delete_resident"):
            st.session_state.flash = f"Deleted {core.delete_memories(sid, rid)} memories of {rid}."
            st.rerun()
    else:
        st.caption("No memories saved in this scenario.")

    st.subheader("Delete")
    sure = st.checkbox("I understand deleting cannot be undone.", key="sure")
    if st.button("Delete all memories in this scenario", key="delete_memories", disabled=not (sure and rows)):
        st.session_state.flash = f"Deleted {core.delete_memories(sid)} memories."
        st.rerun()
    if st.button("Delete this scenario, its report and its memories", key="delete_scenario", disabled=not sure):
        core.delete_scenario(sid)
        st.session_state.flash = f"Deleted scenario {s['name']}."
        st.rerun()

    if len(saved) > 1:
        st.subheader("Compare two scenarios (plan editor stub)")
        for col, default in zip(st.columns(2), (0, 1)):
            with col:
                other = pick_scenario("Scenario", saved, index=default, key=f"compare{default}")
                st.caption(f"Change: {other['settings']['proposed_change'] or 'none'}")
                render_tables(other["report"])


st.set_page_config(page_title="Silicon Town Hall", layout="wide")
gate()
free = st.sidebar.toggle("Use the free test model (no API cost)", value=True, key="free")
cap, vision = 0, False
if not free:
    if config.OPENAI_API_KEY:
        st.sidebar.warning(f"Real model {config.RESIDENT_MODEL}: every call costs money.")
        cap = st.sidebar.number_input("Most real calls per action", min_value=1, max_value=2000, value=50)
        vision = st.sidebar.checkbox("Vision model reranks page images (10 more calls per search)")
    else:
        st.sidebar.warning("No OPENAI_API_KEY is set, so the free test model is used.")
page = st.sidebar.radio("Page", PAGES, key="page")
st.warning(BANNER)
llm = core.make_llm(free, int(cap))
if page == "Community":
    community_page(llm)
elif page == "Policy":
    policy_page()
elif page == "Run meetings":
    run_page(llm, vision)
elif page == "Report":
    report_page()
else:
    saved_page()
