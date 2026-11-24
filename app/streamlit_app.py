"""Silicon Town Hall app (issue #9). Run: streamlit run app/streamlit_app.py

Behind a password; every page carries the simulated banner. The sidebar switch "Use the free test model"
starts ON and then FakeLLM writes every profile, turn and comment, so nothing costs money.
"""

import hmac
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `app` and `townhall` import

import pandas as pd
import streamlit as st

from app import charts, core
from townhall import config

BANNER = "Simulated residents. Not public comment."
PAGES = ["Community", "Policy", "Run meetings", "Report", "Saved scenarios"]
ICONS = {"Community": ":material/groups:", "Policy": ":material/description:", "Run meetings": ":material/forum:",
         "Report": ":material/insights:", "Saved scenarios": ":material/folder_open:"}
TITLES = {"Community": "Choose a community", "Policy": "Choose the policy", "Run meetings": "Run the meetings",
          "Report": "Feedback report", "Saved scenarios": "Saved scenarios"}
BLURBS = {"Community": "Draw residents from real Census records. Each one gets a simulated profile.",
          "Policy": "Pick the plan residents read. Search it the way they will, page by page.",
          "Run meetings": "Residents meet by neighborhood, read the plan, argue and write comments. All simulated.",
          "Report": "What residents raised, where and by whom, how that compares with the real comments, "
                    "and who to reach in person.",
          "Saved scenarios": "Replay a meeting, look at what residents remember, compare runs and delete them."}
DIMS = {"neighborhood": "Neighborhood", "age_band": "Age", "race_ethnicity": "Race and ethnicity",
        "tenure": "Own or rent", "income_band": "Income", "disability": "Disability", "commute": "Commute"}
STANCE = {"support": ("blue", ":material/thumb_up:"), "oppose": ("red", ":material/thumb_down:"),
          "mixed": ("gray", ":material/balance:")}
PRESET_CHOICE, UPLOAD_CHOICE = "I-81 impact statement (preset, already indexed)", "Upload a PDF"
SEARCHES = ["noise during construction", "displacement and relocation", "air quality", "the community grid"]
CSS = """<style>
.sth-hero {margin: 0 0 1.4rem;}
.sth-steps {display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: 1.2rem;}
.sth-step {font-size: .8rem; padding: .25rem .75rem; border-radius: 999px; border: 1px solid rgba(128,128,128,.35);
  opacity: .6; white-space: nowrap;}
.sth-step.done {opacity: 1; border-color: rgba(42,120,214,.6);}
.sth-step.current {opacity: 1; background: #2a78d6; border-color: #2a78d6; color: #fff; font-weight: 600;}
.sth-eyebrow {text-transform: uppercase; letter-spacing: .12em; font-size: .74rem; font-weight: 600; color: #2a78d6;}
.sth-title {font-family: Fraunces, Georgia, serif; font-weight: 700; font-size: clamp(2rem, 3.4vw, 2.9rem);
  line-height: 1.08; margin: .25rem 0 .45rem;}
.sth-sub {opacity: .78; font-size: 1.05rem; max-width: 48rem; line-height: 1.5;}
.sth-brand {font-family: Fraunces, Georgia, serif; font-weight: 700; font-size: 1.45rem; line-height: 1.1;}
.sth-tag {font-size: .8rem; opacity: .7; margin: .2rem 0 .6rem;}
.sth-legend {display: flex; flex-wrap: wrap; gap: 1.1rem; font-size: .85rem; opacity: .85; margin: .4rem 0 .2rem;}
.sth-swatch {display: inline-block; width: .8rem; height: .8rem; border-radius: 3px; margin-right: .35rem;
  vertical-align: -1px;}
</style>"""


def md(text):
    """Plain text for st.markdown: a dollar sign there starts LaTeX."""
    return str(text).replace("$", "\\$")


def tenure(value):
    """Short badge text for a Census tenure value; the table keeps the full wording."""
    return ("Owner" if value.startswith("Owned") else "Renter" if value == "Rented"
            else "Group quarters" if value.startswith("N/A") else md(value))


def dark():
    return getattr(st.context.theme, "type", None) == "dark"


def go(page):
    """Button callback that switches the sidebar page; callbacks run before the radio is drawn."""
    return lambda: st.session_state.update(page=page)


def hero(page):
    i = PAGES.index(page)
    done = [bool(st.session_state.get("residents")), "policy" in st.session_state, "current" in st.session_state,
            False, False]
    steps = "".join(
        f'<span class="sth-step{" current" if j == i else " done" if done[j] else ""}">'
        f'{"&#10003;" if done[j] and j != i else j + 1}&nbsp; {p}</span>' for j, p in enumerate(PAGES))
    st.markdown(f'<div class="sth-hero"><div class="sth-steps">{steps}</div>'
                f'<div class="sth-eyebrow">Step {i + 1} of {len(PAGES)}</div>'
                f'<div class="sth-title" role="heading" aria-level="1">{TITLES[page]}</div>'
                f'<div class="sth-sub">{BLURBS[page]}</div></div>', unsafe_allow_html=True)


def expected_password():
    """st.secrets on Streamlit Cloud, else APP_PASSWORD from .env."""
    if st.secrets.load_if_toml_exists() and "APP_PASSWORD" in st.secrets:
        return str(st.secrets["APP_PASSWORD"])
    return config.APP_PASSWORD


def gate():
    """Stops the script until the right password is typed; nothing else renders before that."""
    if st.session_state.get("authed"):
        return
    _, mid, _ = st.columns([1, 1.5, 1])
    with mid:
        st.space("large")
        st.markdown('<div class="sth-hero"><div class="sth-eyebrow">IST 688 Group 1 planning tool</div>'
                    '<div class="sth-title" role="heading" aria-level="1">Silicon Town Hall</div>'
                    '<div class="sth-sub">Try a draft policy on simulated residents built from Census records, '
                    'and see who it misses, before the public meeting.</div></div>', unsafe_allow_html=True)
        with st.container(border=True):
            expected = expected_password()
            if not expected:
                st.error("APP_PASSWORD is not set, so the app stays locked.")
                st.stop()
            typed = st.text_input("Password", type="password", placeholder="Team password")
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


def show(chart, empty="Nothing to chart yet."):
    if chart is None:
        st.caption(empty)
    else:
        st.altair_chart(chart, width="stretch")


def legend(items):
    st.markdown('<div class="sth-legend">' + "".join(
        f'<span><span class="sth-swatch" style="background:{color}"></span>{label}</span>' for color, label in items)
        + "</div>", unsafe_allow_html=True)


def map_block(residents, rows):
    """Neighborhood map filled by simulated minus expected residents, with its legend and what it leaves out."""
    features = core.neighborhood_shapes()
    if not features:
        return st.caption("The map needs the Syracuse neighborhood boundaries, which the Census step saves under "
                          "data/raw/census/. Run the Community step once with the network on.")
    hood_rows = [r for r in rows if r["dimension"] == "neighborhood"]
    st.pydeck_chart(charts.neighborhood_map(features, hood_rows, dark()), height=470)
    if any(r["expected"] is not None for r in hood_rows):
        under, mid, over = (f"rgb{c}" for c in charts.POLES[dark()])
        legend([(under, "Fewer simulated residents than the Census implies"), (mid, "About as many"), (over, "More")])
    else:
        st.caption("Population shares are not available, so every neighborhood is gray; hover for counts.")
    names = {f["properties"]["NAME"] for f in features}
    off = sum(r["neighborhood"] not in names for r in residents)
    if off:
        st.caption(f"{off} simulated residents live outside the neighborhood map and are not drawn.")


def representation_block(rows, key):
    dims = [d for d in DIMS if any(r["dimension"] == d for r in rows)]
    dim = st.pills("Compare on", dims, default="age_band" if "age_band" in dims else dims[0], format_func=DIMS.get,
                   key=key) or dims[0]
    picked = [{"label": r["group"], "simulated": r["simulated"], "expected": r["expected"]}
              for r in rows if r["dimension"] == dim]
    show(charts.representation(picked))
    if all(r["expected"] is None for r in picked):
        st.caption("Census shares are not available, so only the simulated counts are drawn.")


def resident_cards(residents, limit=24):
    hoods = sorted({r["neighborhood"] for r in residents})
    pick = st.pills("Neighborhood", hoods, selection_mode="multi", key="card_hoods")
    shown = [r for r in residents if not pick or r["neighborhood"] in pick]
    for start in range(0, min(len(shown), limit), 3):
        for col, r in zip(st.columns(3), shown[start:start + 3]):
            with col.container(border=True):
                st.markdown(f"**:material/person: {r['resident_id']}**, {md(r['neighborhood'])}")
                st.markdown(f":gray-badge[{r['age']} years] :gray-badge[{md(r['sex'])}] "
                            f":blue-badge[{tenure(r['tenure'])}]")
                st.caption(md(r.get("profile", "")))
                review = r.get("stereotype_review", {})
                if review.get("flagged") or review.get("keyword_flags"):
                    st.markdown(":orange-badge[:material/flag: Stereotype check flagged this profile]")
    if len(shown) > limit:
        st.caption(f"Showing {limit} of {len(shown)}. The Table tab lists everyone.")


def community_page(llm):
    hero("Community")
    with st.container(border=True):
        place_col, n_col, seed_col, go_col = st.columns([3, 1.3, 1, 1.4], vertical_alignment="bottom")
        place = place_col.selectbox("Community", list(core.PRESETS))
        n = n_col.number_input("Number of residents", min_value=1, max_value=300, value=20)
        seed = seed_col.number_input("Seed", min_value=0, value=0)
        clicked = go_col.button("Sample residents", key="sample", type="primary", icon=":material/casino:",
                                width="stretch")
    if clicked:
        state, puma = core.PRESETS[place]
        with st.spinner("Sampling Census records and writing profiles"):
            residents, meta = core.sample_community(state, puma, int(n), int(seed), llm)
        st.session_state.residents = residents
        st.session_state.community = {"community": place, "state": state, "puma": puma, "residents": int(n),
                                      "seed": int(seed), **meta, "profile_usage": core.usage(llm)}
    residents = st.session_state.get("residents")
    if not residents:
        st.info("Pick the size and seed, then sample. Residents are real Census records with simulated profiles.",
                icon=":material/info:")
        return
    c = st.session_state.community
    shares, _ = core.population(c["state"], c["puma"])
    rows = core.representation(residents, shares)
    missing = [r for r in rows if r["simulated"] == 0 and (r["expected"] or 0) >= 0.5]
    ages = [r["age"] for r in residents]
    cols = st.columns(5)
    cols[0].metric("Simulated residents", len(residents), border=True)
    cols[1].metric("Neighborhoods", len({r["neighborhood"] for r in residents}), border=True)
    cols[2].metric("Median age", f"{statistics.median(ages):g}", border=True)
    cols[3].metric("Renters", f"{sum(r['tenure'] == 'Rented' for r in residents) / len(residents):.0%}", border=True)
    cols[4].metric("Groups with nobody", len(missing) if shares else "not known", border=True,
                   help="Census groups that imply at least half a resident at this size but got no simulated one.")
    st.caption(f"{c['community']}, {c['residents']} residents, seed {c['seed']}. Census: {c['census_source']}. "
               f"Profiles: {residents[0]['profile_written_by']}")
    where, who, people, table = st.tabs([":material/map: Map", ":material/bar_chart: Who they are",
                                         ":material/badge: Residents", ":material/table: Table"])
    with where:
        map_block(residents, rows)
    with who:
        representation_block(rows, key="community_dim")
    with people:
        resident_cards(residents)
    with table:
        st.dataframe(core.resident_table(residents), hide_index=True)


def policy_page():
    hero("Policy")
    policy = st.session_state.get("policy", core.I81)
    choice = st.radio("Policy", [PRESET_CHOICE, UPLOAD_CHOICE], index=int(policy["kind"] == "upload"),
                      horizontal=True)
    if choice == PRESET_CHOICE:
        policy = st.session_state.policy = core.I81
    else:
        uploaded = st.file_uploader("Policy PDF", type="pdf")
        if uploaded and st.button("Index this PDF", key="index", type="primary", icon=":material/upload_file:"):
            try:
                with st.spinner("Indexing every page as text and as a page image"):
                    policy = st.session_state.policy = core.index_upload(uploaded.name, uploaded.getvalue())
            except ValueError as e:
                st.error(str(e))
        for up in core.uploads():
            name_col, del_col = st.columns([5, 1], vertical_alignment="center")
            name_col.markdown(f":material/draft: {md(up['name'])}")
            if del_col.button("Delete", key=f"delete_upload_{up['doc_id']}", icon=":material/delete:"):
                core.delete_upload(up["doc_id"])  # the PDF, its index and its page images
                if policy.get("doc_id") == up["doc_id"]:
                    st.session_state.policy = core.I81
                st.rerun()
        if policy["kind"] != "upload":
            st.info("Upload a PDF and index it. It goes into its own collection, apart from the I-81 index. "
                    "Delete removes the PDF, its index and its page images from this computer.")
            return
    with st.container(border=True):
        name, pages = st.columns([4, 1], vertical_alignment="center")
        name.markdown(f":material/description: **Using:** {md(policy['name'])}")
        pages.metric("Pages indexed", core.page_count(policy["db_dir"]))
    if policy.get("pages_without_text"):
        st.warning(f"{policy['pages_without_text']} pages have no text layer and were not OCRed; residents can "
                   "find them only through page image search.")
    query = st.text_input("Search the policy", placeholder="Ask what a resident would look for",
                          icon=":material/search:")
    suggestion = st.pills("Or try", SEARCHES, key="try")
    query = query or suggestion
    if query:
        for hit in core.get_retriever(policy["db_dir"]).search(query, k=5, mode="both", llm=None):
            with st.container(border=True):
                thumb, text = st.columns([1, 5])
                if Path(hit.image_path).exists():
                    thumb.image(hit.image_path, width=130)
                text.markdown(f"**{hit.file}, page {hit.page}**")
                text.write(md(hit.text[:600]))


def run_page(llm, vision):
    hero("Run meetings")
    residents = st.session_state.get("residents")
    if not residents:
        st.info("Sample residents on the Community page first.", icon=":material/info:")
        st.button("Go to Community", on_click=go("Community"), icon=":material/groups:")
        return
    policy = st.session_state.get("policy", core.I81)
    with st.container(border=True):
        st.markdown(f":material/description: **Policy:** {md(policy['name'])}")
        group_by = st.radio("Meet by", list(core.GROUPINGS), format_func=core.GROUPINGS.get, horizontal=True,
                            key="group_by")
        groups = core.meeting_groups(residents, group_by)
        chosen = st.multiselect("Meetings", list(groups), default=list(groups),
                                format_func=lambda g: f"{g} ({len(groups[g])} residents)")
        rounds_col, name_col = st.columns([1, 3])
        rounds = rounds_col.number_input("Rounds", min_value=1, max_value=5, value=1)
        name = name_col.text_input("Scenario name", value=f"{policy['topic']} test")
        change = st.text_area("Try a change (plan editor stub): describe an edit to the policy. Residents hear it as "
                              "part of the policy; the indexed document is not edited.")
    attending = sum(len(groups[g]) for g in chosen)
    cols = st.columns(4)
    cols[0].metric("Residents attending", f"{attending} of {len(residents)}", border=True)
    cols[1].metric("Meetings", len(chosen) * int(rounds), border=True)
    cols[2].metric("Rounds", int(rounds), border=True)
    cols[3].metric("Text by", "Free test model" if core.written_by(llm) == core.FAKE_LABEL else config.RESIDENT_MODEL,
                   border=True)
    st.caption(f"Text by: {core.written_by(llm)}")
    if st.button("Run meetings", key="run", disabled=not chosen, type="primary", icon=":material/play_arrow:"):
        bar = st.progress(0.0, text="Starting")

        def progress(done, total, rnd, hood):
            bar.progress(done / total, text=f"Round {rnd}: {hood} (meeting {done + 1} of {total})")

        scenario = core.run_scenario(name, residents, policy, chosen, int(rounds), llm, vision=vision, change=change,
                                     community=st.session_state.get("community"), on_progress=progress,
                                     group_by=group_by)
        bar.progress(1.0, text="Done")
        st.session_state.current = scenario["id"]
        if scenario["status"] != "complete":
            st.error(f"The run {scenario['status']}. What finished was saved.")
        st.success(f"Saved scenario {scenario['name']}. Open the Report page.")
        st.button("Open the report", on_click=go("Report"), type="primary", icon=":material/insights:")
        if scenario["usage"]["meetings"]["real_calls"]:
            st.caption(f"Real model use: {scenario['usage']['meetings']}")


def latest_comments(scenario):
    latest = {m["neighborhood"]: m["comments"] for m in scenario["meetings"]}  # as run_scenario reports them
    return [c for comments in latest.values() for c in comments]


def kpis(scenario, report):
    stances = Counter(c.get("stance") for c in latest_comments(scenario))
    topics = len(report["tables"][0]["rows"])
    cols = st.columns(4)
    cols[0].metric("Simulated residents", len(scenario["residents"]), border=True)
    cols[1].metric("Simulated comments", report.get("comments", sum(stances.values())), border=True)
    cols[2].metric("Topics raised", f"{topics} of {len(core.cb.load())}", border=True,
                   help="NYSDOT Appendix A topics that at least one simulated comment raised.")
    cols[3].metric("Support, mixed, oppose", " / ".join(str(stances[s]) for s in ("support", "mixed", "oppose")),
                   border=True, help="Stance of each simulated comment after the meeting.")


def render_pilot_scores(scenario):
    """The recorded I-81 pilot scores, the only place where real comments exist to score against."""
    st.subheader("Pilot scores")
    scores = core.pilot_scores() if scenario["settings"]["policy_kind"] == "preset" else None
    if scores is None:
        return st.caption("Pilot scores exist only for the I-81 pilot, where the real comments are known.")
    st.caption(scores["label"])
    if not scores["scores"]:
        return
    runs = scores["scores"]
    meeting, others = runs.get("meeting", {}), {n: s for n, s in runs.items() if n != "meeting"}
    best = max((n for n in others if others[n].get("topic_coverage") is not None),
               key=lambda n: others[n]["topic_coverage"], default=None)
    if meeting.get("topic_coverage") is not None and best:
        rest = " and ".join(f"{charts.CONDITIONS.get(n, n).lower()} {s['topic_coverage']:.1%}"
                            for n, s in others.items() if s.get("topic_coverage") is not None)
        st.info(f"In the recorded pilot, the meeting's comments raised {meeting['topic_coverage']:.1%} of the topics "
                f"in the real comments. The comparisons raised: {rest}.", icon=":material/lightbulb:")
    left, right = st.columns(2)
    with left:
        st.markdown("**Real topics raised** (higher is closer)")
        show(charts.score_bars(runs, "topic_coverage", "Share of real topics raised", ".0%", dark()))
    with right:
        st.markdown("**Topic mix gap** (0 is the same mix, 1 none in common)")
        show(charts.score_bars(runs, "topic_mix_gap", "Jensen Shannon distance", ".2f", dark()))
    with_counts = [n for n, s in runs.items() if s.get("real_topic_counts")]
    if with_counts:
        st.markdown("**What real commenters raised, and what simulated residents raised**")
        pick = st.segmented_control("Simulated condition", with_counts, default=with_counts[0],
                                    format_func=lambda n: charts.CONDITIONS.get(n, n), key="pilot_condition")
        names = {t["code"]: t["topic"] for t in core.cb.load()}
        show(charts.real_vs_simulated(runs[pick or with_counts[0]], names))
    with st.expander("Table view and definitions"):
        pct = lambda v: "pending" if v is None else f"{v:.1%}"
        st.dataframe(pd.DataFrame([{"condition": name.replace("_", " "), "real topics raised": pct(s.get("topic_coverage")),
                                    "topic mix gap": s.get("topic_mix_gap")} for name, s in runs.items()]),
                     hide_index=True)
        st.caption("Real topics raised: share of the topics in the real comments that at least one simulated comment "
                   "raised. Topic mix gap: 0 is the same mix of topics, 1 is no topic in common. Conditions: the "
                   "meeting; one model with no personas; the same residents writing without the meeting.")


def render_reach(scenario, report):
    st.subheader("Who to reach in person")
    reach = report.get("reach")
    if not report["reach_in_person"]:
        return st.success("Nobody flagged.", icon=":material/check_circle:")
    if reach is None:  # scenarios saved before the charts carry only the sentences
        return st.markdown("\n".join(f"- {md(item)}" for item in report["reach_in_person"]))
    chart, where, sentences = st.tabs([":material/scatter_plot: Shortfall", ":material/map: Map",
                                       ":material/list: Reasons"])
    with chart:
        top = reach[:12]
        show(charts.representation([{"label": f"{DIMS.get(e['dimension'], e['dimension'])}: {e['group']}",
                                     "simulated": e["simulated_residents"], "expected": e["expected_residents"]}
                                    for e in top]))
        st.caption(f"The {len(top)} biggest shortfalls of {len(reach)} flagged groups; the Reasons tab lists all.")
    with where:
        settings = scenario["settings"]
        if "state" in settings:
            shares, _ = core.population(settings["state"], settings["puma"])
            map_block(scenario["residents"], core.representation(scenario["residents"], shares))
        else:
            st.caption("This scenario has no community settings to map.")
    with sentences:
        st.markdown("\n".join(f"- {md(item)}" for item in report["reach_in_person"]))


def render_topics(report):
    rows = report["tables"]
    st.subheader("Topics raised")
    show(charts.topic_bars(rows[0]["rows"], dark()), "No simulated comment raised a codebook topic.")
    with st.expander("Table view"):
        st.dataframe(pd.DataFrame(rows[0]["rows"]), hide_index=True)
    matrix = report.get("matrix")
    st.subheader("Topics by neighborhood")
    if matrix is not None:
        show(charts.heatmap([m for m in matrix if m["dimension"] == "neighborhood"], dark()))
    with st.expander("Table view"):
        st.dataframe(pd.DataFrame(rows[1]["rows"]), hide_index=True)
    st.subheader("Topics by group")
    if matrix is not None:
        dims = [d for d in DIMS if d != "neighborhood" and any(m["dimension"] == d for m in matrix)]
        if dims:
            dim = st.pills("Group by", dims, default=dims[0], format_func=DIMS.get, key="group_dim") or dims[0]
            show(charts.heatmap([m for m in matrix if m["dimension"] == dim], dark()))
    with st.expander("Table view"):
        st.dataframe(pd.DataFrame(rows[2]["rows"]), hide_index=True)


def render_samples(report):
    st.subheader("Sample comments (SIMULATED)")
    merged = {}  # the analyst samples per topic, so one comment can come back under several topics
    for c in report["sample_comments"]:
        merged.setdefault((c["resident_id"], c["text"]), {**c, "topics": []})["topics"].append(c.get("topic"))
    topics = sorted({t for c in merged.values() for t in c["topics"] if t})
    pick = st.pills("Filter by topic", topics, selection_mode="multi", key="sample_topics")
    for c in merged.values():
        if pick and not set(pick) & set(c["topics"]):
            continue
        color, icon = STANCE.get(c["stance"], ("gray", ":material/chat:"))
        with st.chat_message(c["resident_id"], avatar=icon):
            st.markdown(f"**SIMULATED** comment by {c['resident_id']}, {md(c['neighborhood'])} "
                        f":{color}-badge[{c['stance']}] " + " ".join(f":violet-badge[topic: {md(t)}]"
                                                                     for t in c["topics"] if t))
            st.markdown(md(c["text"]))
            st.caption(f"Cited: {c['cited']}")


def render_report(scenario):
    report = scenario["report"]
    if scenario["status"] != "complete":
        st.error(f"This run {scenario['status']}")
    if report is None:
        return st.error("This scenario has no report: "
                        f"{scenario.get('report_error', 'building it failed after the meetings were saved')}.")
    st.caption(report["label"])
    kpis(scenario, report)
    render_pilot_scores(scenario)
    render_reach(scenario, report)
    render_topics(report)
    render_samples(report)


def report_page():
    hero("Report")
    saved = core.list_scenarios()
    if not saved:
        st.info("Run the meetings first.", icon=":material/info:")
        return
    ids = [s["id"] for s in saved]
    current = st.session_state.get("current")
    render_report(pick_scenario("Scenario", saved, index=ids.index(current) if current in ids else 0))


def replay(s):
    """A meeting transcript as a chat, one bubble per turn, avatar by stance."""
    meetings = s["meetings"]
    if not meetings:
        return st.caption("No meeting finished in this scenario.")
    labels = [f"Round {m['round']}, {m['neighborhood']}" for m in meetings]
    i = labels.index(st.selectbox("Meeting", labels, key="replay_meeting")) if len(meetings) > 1 else 0
    m, by_id = meetings[i], {r["resident_id"]: r for r in s["residents"]}
    agenda, summary = st.columns([3, 2])
    with agenda.container(border=True):
        st.markdown("**Agenda** (SIMULATED)")
        st.markdown("\n".join(f"{n}. {md(item)}" for n, item in enumerate(m["agenda"], 1)))
    with summary.container(border=True):
        st.markdown("**Summary** (SIMULATED)")
        st.caption(md(m["summary"]))
    items = list(dict.fromkeys(t.get("agenda_item") for t in m["transcript"] if t.get("agenda_item")))
    pick = st.pills("Agenda item", items, key=f"replay_item_{i}") if len(items) > 1 else None
    for t in m["transcript"]:
        if pick and t.get("agenda_item") != pick:
            continue
        r = by_id.get(t["resident_id"], {})
        color, icon = STANCE.get(t["stance"], ("gray", ":material/chat:"))
        with st.chat_message(t["resident_id"], avatar=icon):
            who = ", ".join(x for x in (r.get("age") and f"{r['age']} years", r.get("tenure") and tenure(r["tenure"])) if x)
            st.markdown(f"**{t['resident_id']}**, {md(t['neighborhood'])}" + (f", {who}" if who else "")
                        + f" :{color}-badge[{t['stance']}] :gray-badge[SIMULATED]")
            st.markdown(md(t["text"]))
            st.caption(f"Cited: {core.cite(t['cited_pages'])}")
    with st.expander("Table view"):
        st.dataframe(pd.DataFrame([{"label": "SIMULATED", "resident_id": t["resident_id"], "stance": t["stance"],
                                    "text": t["text"], "cited": core.cite(t["cited_pages"])} for t in m["transcript"]]),
                     hide_index=True)


def saved_page():
    hero("Saved scenarios")
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))
    saved = core.list_scenarios()
    if not saved:
        st.info("No saved scenarios.", icon=":material/info:")
        return
    st.dataframe(pd.DataFrame([{"name": s["name"], "date": s["date"], "policy": s["settings"]["policy"],
                                "residents": len(s["residents"]), "meetings": len(s["meetings"]),
                                "rounds": s["settings"]["rounds"], "text by": s["settings"]["meeting_text_by"],
                                "status": s["status"]} for s in saved]), hide_index=True)
    s = pick_scenario("View a scenario", saved, key="view")
    sid = s["id"]
    watch, remember, settings, remove = st.tabs([":material/forum: Meeting replay", ":material/psychology: Memories",
                                                 ":material/tune: Settings", ":material/delete: Delete"])
    with watch:
        replay(s)
    with settings:
        st.json(s["settings"])
    with remember:
        st.caption("What each simulated resident carries into the next round (SIMULATED).")
        rows = core.memories(sid)
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            rid = st.selectbox("Resident", sorted({r["resident_id"] for r in rows}), key="memory_resident")
            if st.button("Delete this resident's memories", key="delete_resident", icon=":material/person_remove:"):
                st.session_state.flash = f"Deleted {core.delete_memories(sid, rid)} memories of {rid}."
                st.rerun()
        else:
            st.caption("No memories saved in this scenario.")
    with remove, st.container(border=True):
        st.markdown(":material/warning: **Deleting cannot be undone.** Memories are removed with secure delete.")
        sure = st.checkbox("I understand deleting cannot be undone.", key="sure")
        if st.button("Delete all memories in this scenario", key="delete_memories", disabled=not (sure and rows)):
            st.session_state.flash = f"Deleted {core.delete_memories(sid)} memories."
            st.rerun()
        if st.button("Delete this scenario, its report and its memories", key="delete_scenario", disabled=not sure,
                     type="primary"):
            core.delete_scenario(sid)
            st.session_state.flash = f"Deleted scenario {s['name']}."
            st.rerun()

    if len(saved) > 1:
        st.subheader("Compare two scenarios (plan editor stub)")
        for col, default in zip(st.columns(2), (0, 1)):
            with col, st.container(border=True):
                other = pick_scenario("Scenario", saved, index=default, key=f"compare{default}")
                st.caption(f"Change: {other['settings']['proposed_change'] or 'none'}")
                stances = Counter(c.get("stance") for c in latest_comments(other))
                st.metric("Support, mixed, oppose, undecided",
                          " / ".join(str(stances[s]) for s in ("support", "mixed", "oppose", "undecided")))
                if other["report"] is None:
                    st.error("This scenario has no report: building it failed after the meetings were saved.")
                else:
                    show(charts.topic_bars(other["report"]["tables"][0]["rows"], dark()),
                         "No simulated comment raised a codebook topic.")


st.set_page_config(page_title="Silicon Town Hall", page_icon=":material/account_balance:", layout="wide")
st.html(CSS)
gate()
with st.sidebar:
    st.markdown('<div class="sth-brand">Silicon Town Hall</div>'
                '<div class="sth-tag">Simulated residents for planning teams</div>', unsafe_allow_html=True)
    free = st.toggle("Use the free test model (no API cost)", value=True, key="free")
    cap, vision = 0, False
    if not free:
        if config.OPENAI_API_KEY:
            st.warning(f"Real model {config.RESIDENT_MODEL}: every call costs money.")
            cap = st.number_input("Most real calls per action", min_value=1, max_value=2000, value=50)
            vision = st.checkbox("Vision model reranks page images (10 more calls per search)")
        else:
            st.warning("No OPENAI_API_KEY is set, so the free test model is used.")
    page = st.radio("Page", PAGES, key="page", format_func=lambda p: f"{ICONS[p]} {p}")
    status = st.container(border=True)  # filled after the page runs, so a click on this run shows up in it
st.warning(BANNER, icon=":material/smart_toy:")
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
with status:
    st.caption("This session")
    st.markdown(f":material/groups: {len(st.session_state.get('residents') or [])} residents sampled  \n"
                f":material/description: {md(st.session_state.get('policy', core.I81)['topic'])}  \n"
                f":material/folder_open: {len(core.list_scenarios())} saved scenarios")
