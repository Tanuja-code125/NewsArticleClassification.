import html

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Story Sorter", page_icon="🗞️", layout="wide")

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&display=swap');

:root {
    --ink: #0E1B33;
    --muted: #5B6678;
    --paper: #EDF0F4;
    --panel: #FFFFFF;
    --line: #D3D9E3;
    --bar: #B9C3D4;
    --marker: #FFD84D;
}

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
header[data-testid="stHeader"] { background: transparent; }

.stApp { background: var(--paper); }
.block-container { max-width: 1080px; padding-top: 3.5rem; padding-bottom: 4rem; }

html, body, .stApp, .stApp p, .stApp label, .stApp button {
    font-family: 'Bricolage Grotesque', sans-serif;
    color: var(--ink);
}

/* heading */
.title {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    line-height: 1.05;
    letter-spacing: -0.03em;
    margin: 0 0 0.9rem 0;
    max-width: 14em;
}
.sub {
    font-size: 1.1rem;
    color: var(--muted);
    max-width: 38em;
    margin: 0 0 2.4rem 0;
}

/* article box */
.stTextArea textarea {
    font-family: 'Newsreader', serif;
    font-size: 1.2rem;
    line-height: 1.6;
    background: var(--panel);
    color: var(--ink);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 1.2rem 1.3rem;
    box-shadow: none;
}
.stTextArea textarea::placeholder { color: #8A94A6; font-style: italic; }
.stTextArea textarea:focus {
    border-color: var(--ink);
    box-shadow: 0 0 0 3px rgba(255, 216, 77, 0.55);
}
.stTextArea [data-baseweb="textarea"],
.stTextArea [data-baseweb="base-input"] { background: transparent; border: none; }

/* buttons */
button[data-testid="stBaseButton-primary"] {
    background: var(--ink);
    color: #fff;
    border: none;
    border-radius: 4px;
    font-weight: 600;
    font-size: 1.05rem;
    padding: 0.55rem 1.6rem;
}
button[data-testid="stBaseButton-primary"] p { color: #fff; }
button[data-testid="stBaseButton-primary"]:hover { background: #1D3157; }
button[data-testid="stBaseButton-primary"]:focus-visible { outline: 3px solid var(--marker); }

button[data-testid="stBaseButton-secondary"] {
    background: transparent;
    border: 1px solid var(--line);
    border-radius: 999px;
    font-size: 0.92rem;
    padding: 0.15rem 0.9rem;
    min-height: 0;
}
button[data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--ink);
    background: var(--marker);
}

.hint { color: var(--muted); font-size: 0.95rem; margin: 1.1rem 0 0.4rem 0; }
.count { color: var(--muted); font-size: 0.9rem; margin-top: -0.4rem; }

/* result */
.result {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 1.6rem 1.6rem 1.4rem 1.6rem;
}
.result .small { color: var(--muted); font-size: 0.95rem; margin: 0 0 0.3rem 0; }
.result .answer {
    font-family: 'Newsreader', serif;
    font-weight: 600;
    font-size: 2.6rem;
    line-height: 1.1;
    margin: 0 0 0.5rem 0;
    display: inline;
    background: linear-gradient(transparent 58%, var(--marker) 58%);
}
.result .conf { margin: 0.9rem 0 1.4rem 0; color: var(--muted); font-size: 1rem; }
.result .conf b { color: var(--ink); font-weight: 600; }
.result .note { margin: 0 0 1.4rem 0; font-size: 0.98rem; }

.row { display: grid; grid-template-columns: 8.5rem 1fr 3rem; align-items: center; gap: 0.7rem; margin: 0.55rem 0; font-size: 0.95rem; }
.row .lbl { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.row .track { background: #EEF1F6; height: 0.6rem; border-radius: 2px; }
.row .fill { background: var(--bar); height: 100%; border-radius: 2px; }
.row.top .fill { background: var(--ink); }
.row .pct { text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }

.empty {
    border: 1.5px dashed var(--line);
    border-radius: 4px;
    padding: 2rem 1.6rem;
    color: var(--muted);
    font-size: 1.05rem;
}
.empty b { color: var(--ink); font-weight: 600; }

.foot { color: var(--muted); font-size: 0.88rem; margin-top: 3rem; }

@media (max-width: 700px) {
    .title { font-size: 2.2rem; }
    .row { grid-template-columns: 6.5rem 1fr 2.6rem; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


model = load_model()
n_sections = len(model.classes_)


def nice(label: str) -> str:
    return label.strip().title()


def score(text: str) -> pd.Series:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba([text])[0]
    else:  # LinearSVC has no probabilities, so turn its scores into a rough share
        s = model.decision_function([text])[0] * 5
        e = np.exp(s - s.max())
        p = e / e.sum()
    out = pd.Series(p, index=[nice(c) for c in model.classes_])
    return out.sort_values(ascending=False)


SAMPLES = {
    "Sports": "The captain scored a late winner in extra time as the home side beat their rivals 2-1 and booked a place in the final.",
    "Politics": "The Senate voted late Thursday to pass a bipartisan spending bill, sending it to the president's desk hours before funding ran out.",
    "Travel": "This quiet coastal town has empty beaches, fresh seafood and cheap guesthouses, and it is the best place to visit this summer.",
    "Food": "A simple one-pan recipe for crispy roast chicken with lemon and garlic that takes ten minutes to prepare.",
}


def use_sample(name: str):
    st.session_state["story"] = SAMPLES[name]
    st.session_state["result"] = None


if "result" not in st.session_state:
    st.session_state["result"] = None

# ---------------------------------------------------------------- header
st.markdown(
    f"""
<h1 class="title">Which section does this story belong in?</h1>
<p class="sub">Paste a headline or a full article. The model reads it and files it under one of {n_sections} news sections.</p>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([3, 2], gap="large")

# ---------------------------------------------------------------- input
with left:
    story = st.text_area(
        "Story",
        key="story",
        height=300,
        placeholder="Paste the story here...",
        label_visibility="collapsed",
    )
    words = len(story.split())
    st.markdown(f'<div class="count">{words} words</div>', unsafe_allow_html=True)

    if st.button("Classify story", type="primary"):
        if words < 5:
            st.session_state["result"] = "short"
        else:
            st.session_state["result"] = score(story)

    st.markdown('<div class="hint">No story handy? Try one of these.</div>', unsafe_allow_html=True)
    cols = st.columns(len(SAMPLES) + 2)
    for col, name in zip(cols, SAMPLES):
        col.button(name, key=f"s_{name}", on_click=use_sample, args=(name,))

# ---------------------------------------------------------------- result
with right:
    res = st.session_state["result"]

    if res is None:
        st.markdown(
            '<div class="empty"><b>Your result shows up here.</b><br>'
            "Paste a story on the left and press Classify story.</div>",
            unsafe_allow_html=True,
        )
    elif isinstance(res, str):
        st.markdown(
            '<div class="empty"><b>That is too short to sort.</b><br>'
            "Add at least five words, such as a full headline.</div>",
            unsafe_allow_html=True,
        )
    else:
        top = res.head(5)
        label, conf = top.index[0], top.iloc[0]

        note = ""
        if conf - top.iloc[1] < 0.08:
            note = (
                f'<p class="note">This one is a close call between '
                f"<b>{html.escape(top.index[0])}</b> and <b>{html.escape(top.index[1])}</b>.</p>"
            )

        rows = ""
        for i, (name, p) in enumerate(top.items()):
            cls = "row top" if i == 0 else "row"
            rows += (
                f'<div class="{cls}"><div class="lbl">{html.escape(name)}</div>'
                f'<div class="track"><div class="fill" style="width:{p * 100:.0f}%"></div></div>'
                f'<div class="pct">{p:.0%}</div></div>'
            )

        st.markdown(
            f'<div class="result"><p class="small">Best fit</p>'
            f'<div class="answer">{html.escape(label)}</div>'
            f'<p class="conf">Confidence <b>{conf:.0%}</b></p>'
            f"{note}{rows}</div>",
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="foot">Trained on about 200,000 news headlines. On stories it had not seen before, '
    "it picked the right section about 63% of the time.</div>",
    unsafe_allow_html=True,
)