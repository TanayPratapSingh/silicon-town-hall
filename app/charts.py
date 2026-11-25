"""Charts for the app: Altair for bars, dots and heatmaps, pydeck for the neighborhood map. Pure functions of
plain rows, so tests build them without Streamlit. Both libraries ship with Streamlit.

Colors are the dataviz reference palette: blue is the series that matters, gray is the reference or the
baselines, and the map is diverging (red fewer residents than the Census implies, blue more, gray about right).
"""

import altair as alt
import pandas as pd
import pydeck as pdk

BLUE, GRAY = "#2a78d6", "#898781"
REAL, SIM = "Real comments (NYSDOT)", "Simulated comments"
EXPECTED, SIMULATED = "Census expected", "Simulated residents"
CONDITIONS = {"meeting": "Meeting (personas and debate)", "no_personas": "One model, no personas",
              "skip_meeting": "Personas, no meeting"}
POLES = ((227, 73, 72), (240, 239, 236), (42, 120, 214))  # under, about right, over
INK = "#4a5562"  # secondary text ink; values never wear the series color
RAMP = ["#b7d3f6", "#0d366b"]  # low to high: the low end recedes into the white surface


def _height(rows, per=28, pad=70):
    """Streamlit fits the whole chart, legend and axis included, into this height, so pad leaves them room."""
    return per * max(rows, 1) + pad


def topic_bars(rows):
    """rows: [{topic, code, simulated comments}] -> one series, ranked."""
    df = pd.DataFrame([{"topic": r["topic"], "code": r["code"], "n": r["simulated comments"]} for r in rows])
    if df.empty:
        return None
    base = alt.Chart(df).encode(
        y=alt.Y("topic:N", sort="-x", title=None, axis=alt.Axis(labelLimit=280)),
        x=alt.X("n:Q", title="Simulated comments that raise the topic", axis=alt.Axis(tickMinStep=1)),
        tooltip=[alt.Tooltip("topic:N", title="Topic"), alt.Tooltip("code:N", title="NYSDOT code"),
                 alt.Tooltip("n:Q", title="Simulated comments")])
    bars = base.mark_bar(color=BLUE, cornerRadiusEnd=4, size=16)
    return (bars + base.mark_text(align="left", dx=5, color=INK).encode(text="n:Q")).properties(
        height=_height(len(df)))


def real_vs_simulated(score, names):
    """Share of real and of simulated comments that raise each topic, topics ranked by the real share.
    None when the score file has no topic counts."""
    real, sim = score.get("real_topic_counts"), score.get("simulated_topic_counts")
    if not (real and sim is not None and score.get("real_comments") and score.get("simulated_comments")):
        return None
    rows = []
    for code in set(real) | set(sim):
        for source, counts, total in ((REAL, real, score["real_comments"]), (SIM, sim, score["simulated_comments"])):
            rows.append({"topic": names.get(code, code), "source": source, "n": counts.get(code, 0),
                         "share": counts.get(code, 0) / total, "total": total})
    df = pd.DataFrame(rows)
    order = df[df.source == REAL].sort_values("share", ascending=False).topic.tolist()
    return alt.Chart(df).mark_bar(cornerRadiusEnd=4, size=9).encode(
        y=alt.Y("topic:N", sort=order, title=None, axis=alt.Axis(labelLimit=280)),
        yOffset=alt.YOffset("source:N", sort=[REAL, SIM]),
        x=alt.X("share:Q", title="Share of comments that raise the topic", axis=alt.Axis(format="%")),
        color=alt.Color("source:N", scale=alt.Scale(domain=[REAL, SIM], range=[GRAY, BLUE]),
                        legend=alt.Legend(orient="top", title=None, labelLimit=0)),
        tooltip=[alt.Tooltip("topic:N", title="Topic"), alt.Tooltip("source:N", title="Comments"),
                 alt.Tooltip("share:Q", title="Share", format=".1%"), alt.Tooltip("n:Q", title="Comments raising it"),
                 alt.Tooltip("total:Q", title="Comments in all")],
    ).properties(height=_height(len(order), per=34, pad=110))


def score_bars(scores, key, title, fmt):
    """One measure across the conditions; the meeting is blue, the baselines gray, the value at each bar tip."""
    df = pd.DataFrame([{"condition": CONDITIONS.get(name, name), "value": s[key], "meeting": name == "meeting"}
                       for name, s in scores.items() if s.get(key) is not None])
    if df.empty:
        return None
    base = alt.Chart(df).encode(
        y=alt.Y("condition:N", sort=None, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("value:Q", title=title, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=fmt)),
        tooltip=[alt.Tooltip("condition:N", title="Condition"), alt.Tooltip("value:Q", title=title, format=fmt)])
    bars = base.mark_bar(cornerRadiusEnd=4, size=18).encode(
        color=alt.condition("datum.meeting", alt.value(BLUE), alt.value(GRAY)))
    return (bars + base.mark_text(align="left", dx=5, color=INK).encode(
        text=alt.Text("value:Q", format=fmt))).properties(
        height=_height(len(df), per=34))


def representation(rows):
    """Dumbbell: simulated residents against the Census expected count per group. rows: [{label, simulated, expected}]."""
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    order = df.label.tolist()
    dots = pd.concat([df.assign(series=SIMULATED, residents=df.simulated),
                      df[df.expected.notna()].assign(series=EXPECTED, residents=df.expected)])
    y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
    tooltip = [alt.Tooltip("label:N", title="Group"), alt.Tooltip("simulated:Q", title=SIMULATED),
               alt.Tooltip("expected:Q", title=EXPECTED, format=".1f")]
    rule = alt.Chart(df[df.expected.notna()]).mark_rule(color=GRAY, strokeWidth=2, opacity=0.6).encode(
        y=y, x="simulated:Q", x2="expected:Q", tooltip=tooltip)
    points = alt.Chart(dots).mark_circle(size=120, opacity=1).encode(
        y=y, x=alt.X("residents:Q", title="Residents", axis=alt.Axis(tickMinStep=1)),
        color=alt.Color("series:N", scale=alt.Scale(domain=[EXPECTED, SIMULATED], range=[GRAY, BLUE]),
                        legend=alt.Legend(orient="top", title=None, labelLimit=0)),
        tooltip=tooltip)
    return (rule + points).properties(height=_height(len(df), per=32, pad=110))


def heatmap(rows):
    """Topic by group counts, sequential blue; empty cells mean no simulated comment. rows: [{group, topic, comments}]."""
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    topics = df.groupby("topic").comments.sum().sort_values(ascending=False).index.tolist()
    return alt.Chart(df).mark_rect(cornerRadius=3).encode(
        x=alt.X("group:N", title=None, axis=alt.Axis(labelAngle=-40, labelLimit=170, orient="top",
                                                    labelOverlap=False)),
        y=alt.Y("topic:N", sort=topics, title=None, axis=alt.Axis(labelLimit=280)),
        color=alt.Color("comments:Q", title="Simulated comments", scale=alt.Scale(range=RAMP),
                        legend=alt.Legend(orient="bottom", tickMinStep=1)),
        tooltip=[alt.Tooltip("group:N", title="Group"), alt.Tooltip("topic:N", title="Topic"),
                 alt.Tooltip("comments:Q", title="Simulated comments")],
    ).properties(height=_height(len(topics), per=30, pad=230))


def gap_color(gap, span=3.0):
    """Diverging fill for simulated minus expected residents: red under, gray about right, blue over."""
    under, mid, over = POLES
    if gap is None:
        return [*mid, 120]
    t = max(-1.0, min(1.0, gap / span))
    pole = over if t > 0 else under
    return [round(m + (p - m) * abs(t)) for m, p in zip(mid, pole)] + [200]


def neighborhood_map(features, rows):
    """pydeck map of the neighborhood polygons, filled by simulated minus expected residents.
    rows: representation rows for the neighborhood dimension, [{group, simulated, expected}]."""
    by = {r["group"]: r for r in rows}
    shaped, xs, ys = [], [], []
    for f in features:
        name = f["properties"]["NAME"]
        r = by.get(name, {"simulated": 0, "expected": None})
        gap = None if r["expected"] is None else r["simulated"] - r["expected"]
        shaped.append({"type": "Feature", "geometry": f["geometry"], "properties": {
            "name": name, "simulated": r["simulated"],
            "expected": "not known" if r["expected"] is None else f"{r['expected']:.1f}",
            "fill": gap_color(gap)}})
        rings = f["geometry"]["coordinates"] if f["geometry"]["type"] == "Polygon" else [
            ring for poly in f["geometry"]["coordinates"] for ring in poly]
        for ring in rings:
            xs += [p[0] for p in ring]
            ys += [p[1] for p in ring]
    layer = pdk.Layer("GeoJsonLayer", {"type": "FeatureCollection", "features": shaped}, pickable=True,
                      auto_highlight=True, stroked=True, filled=True, get_fill_color="properties.fill",
                      get_line_color=[255, 255, 255, 220], line_width_min_pixels=1.5)
    view = pdk.ViewState(longitude=(min(xs) + max(xs)) / 2, latitude=(min(ys) + max(ys)) / 2, zoom=11.6)
    tooltip = {"html": "<b>{name}</b><br/>{simulated} simulated residents<br/>{expected} expected from the Census",
               "style": {"fontFamily": "EB Garamond, Garamond, serif", "fontSize": "15px"}}
    # no scroll zoom: the wheel scrolls the page past the map; the + and - buttons and drag still work
    return pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip, map_style=None,
                    views=[pdk.View(type="MapView", controller={"scrollZoom": False, "dragRotate": False})])
