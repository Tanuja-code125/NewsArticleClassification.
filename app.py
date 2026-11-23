import html
import re
import sys
import warnings
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ============================================================ Config
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MODEL_PATH = RESULTS_DIR / "best_news_model.joblib"
METADATA_PATH = RESULTS_DIR / "model_metadata.joblib"
COMPARISON_CSV = RESULTS_DIR / "model_comparison.csv"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from preprocessing import clean_text  # noqa: E402

st.set_page_config(
    page_title="Story Sorter — News Classifier",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def h(s: str) -> str:
    """Flatten HTML so Streamlit's markdown never treats indentation as a code block."""
    return "".join(line.strip() for line in s.splitlines())


# ============================================================ Styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700;9..144,800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
:root{
  --bg:#0B0F17; --surface:#121826; --surface2:#182034; --line:#26304A;
  --text:#E8ECF4; --muted:#8B96AE; --accent:#FFB020; --accent2:#5B8CFF;
  --good:#34D399; --warn:#FBBF24;
}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{display:none;}
header[data-testid="stHeader"]{background:transparent;}
.stApp{
  background:
    radial-gradient(900px 400px at 85% -10%, rgba(91,140,255,.14), transparent 60%),
    radial-gradient(700px 380px at 0% 0%, rgba(255,176,32,.09), transparent 60%),
    var(--bg);
  color:var(--text); font-family:'Inter',sans-serif;
}
html, body, .stApp, .stApp p, .stApp label, .stApp span, .stApp li{font-family:'Inter',sans-serif;}
.stApp p, .stApp label, .stApp li{color:var(--text);}
.block-container{max-width:1200px; padding-top:1.6rem; padding-bottom:4rem;}
h1,h2,h3,h4,h5{font-family:'Fraunces',serif !important; color:var(--text) !important; letter-spacing:-.01em;}
[data-testid="stCaptionContainer"], .stCaption{color:var(--muted) !important;}
hr{border-color:var(--line) !important;}

/* ---------- Hero */
.hero{display:flex; justify-content:space-between; align-items:flex-end; gap:2rem; flex-wrap:wrap;
      padding:.6rem 0 1.6rem 0; border-bottom:1px solid var(--line); margin-bottom:1.4rem;}
.eyebrow{display:inline-flex; align-items:center; gap:.5rem; font:600 .72rem 'JetBrains Mono',monospace;
         letter-spacing:.16em; text-transform:uppercase; color:var(--accent);}
.eyebrow::before{content:""; width:.55rem; height:.55rem; border-radius:50%; background:var(--accent);
                 box-shadow:0 0 0 4px rgba(255,176,32,.2);}
.hero-title{font-family:'Fraunces',serif; font-weight:800; font-size:3.1rem; line-height:1.02;
            letter-spacing:-.03em; margin:.55rem 0 .6rem 0; color:var(--text);}
.hero-title em{font-style:italic; color:var(--accent); font-weight:700;}
.hero-sub{color:var(--muted); font-size:1.02rem; max-width:34rem; line-height:1.55; margin:0;}
.hero-stats{display:flex; gap:.8rem; flex-wrap:wrap;}
.stat{background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:.8rem 1.1rem; min-width:7.2rem;}
.stat b{display:block; font:800 1.55rem 'Fraunces',serif; color:var(--text);}
.stat span{font:600 .68rem 'JetBrains Mono',monospace; letter-spacing:.1em; text-transform:uppercase; color:var(--muted);}

/* ---------- Tabs */
.stTabs [data-baseweb="tab-list"]{gap:.3rem; border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{height:2.8rem; padding:0 1.1rem; background:transparent; color:var(--muted);
                             font-weight:600; border-radius:8px 8px 0 0;}
.stTabs [aria-selected="true"]{color:var(--text) !important; background:var(--surface);}
.stTabs [data-baseweb="tab-highlight"]{background:var(--accent) !important; height:3px;}
.stTabs [data-baseweb="tab-border"]{display:none;}

/* ---------- Panels & KPI */
.panel{background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:1.4rem;}
.section-label{font:600 .72rem 'JetBrains Mono',monospace; letter-spacing:.14em; text-transform:uppercase;
               color:var(--muted); margin:0 0 .7rem 0;}
.kpi{background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:1.1rem 1.2rem; height:100%;}
.kpi .l{font:600 .68rem 'JetBrains Mono',monospace; letter-spacing:.12em; text-transform:uppercase; color:var(--muted);}
.kpi .v{font:800 2rem 'Fraunces',serif; margin-top:.25rem; color:var(--text); line-height:1.15;}
.kpi .s{font-size:.8rem; color:var(--muted); margin-top:.15rem;}

/* ---------- Inputs */
[data-testid="stForm"]{background:var(--surface); border:1px solid var(--line) !important; border-radius:16px; padding:1.3rem;}
.stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div{
  background:var(--surface2) !important; color:var(--text) !important;
  border:1px solid var(--line) !important; border-radius:10px !important;}
.stTextArea textarea{font-family:'Inter',sans-serif; font-size:1.02rem; line-height:1.6;}
.stTextInput input:focus, .stTextArea textarea:focus{border-color:var(--accent) !important; box-shadow:0 0 0 3px rgba(255,176,32,.18) !important;}
.stTextInput label, .stTextArea label{color:var(--muted) !important; font-weight:600; font-size:.82rem;}

button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"]{
  background:var(--accent); color:#1A1204; border:none; border-radius:10px; font-weight:700; padding:.6rem 1.2rem;
  transition:transform .12s ease, filter .12s ease;}
button[data-testid="stBaseButton-primary"]:hover, button[data-testid="stBaseButton-primaryFormSubmit"]:hover{filter:brightness(1.08); transform:translateY(-1px);}
button[data-testid="stBaseButton-secondary"], button[data-testid="stBaseButton-secondaryFormSubmit"]{
  background:var(--surface2); color:var(--text); border:1px solid var(--line); border-radius:999px; font-weight:600; font-size:.85rem;}
button[data-testid="stBaseButton-secondary"]:hover, button[data-testid="stBaseButton-secondaryFormSubmit"]:hover{border-color:var(--accent); color:var(--accent);}

/* ---------- Result */
.result{background:linear-gradient(180deg, var(--surface), var(--surface2)); border:1px solid var(--line);
        border-radius:18px; padding:1.5rem; animation:rise .35s ease;}
@keyframes rise{from{opacity:0; transform:translateY(8px);} to{opacity:1; transform:none;}}
.result-top{display:flex; justify-content:space-between; align-items:center; gap:1rem;}
.cat-name{font:800 2.1rem 'Fraunces',serif; line-height:1.1; letter-spacing:-.02em; margin:.2rem 0 .3rem 0;}
.cat-desc{color:var(--muted); font-size:.92rem; margin:0;}
.ring{--p:0; --c:#FFB020; width:5.6rem; height:5.6rem; border-radius:50%; flex:none;
      background:conic-gradient(var(--c) calc(var(--p)*1%), #26304A 0); display:grid; place-items:center;}
.ring div{width:4.5rem; height:4.5rem; border-radius:50%; background:var(--surface);
          display:grid; place-items:center; font:800 1.15rem 'Fraunces',serif; color:var(--text);}
.badge{display:inline-block; font:600 .68rem 'JetBrains Mono',monospace; letter-spacing:.1em; text-transform:uppercase;
       padding:.2rem .6rem; border-radius:999px; border:1px solid var(--line); color:var(--muted); margin-right:.4rem;}
.callout{background:rgba(251,191,36,.08); border:1px solid rgba(251,191,36,.35); color:#FDE68A;
         border-radius:10px; padding:.7rem .9rem; font-size:.9rem; margin:1rem 0 0 0;}
.prob{display:grid; grid-template-columns:9.5rem 1fr 3.4rem; align-items:center; gap:.7rem; margin:.5rem 0; font-size:.88rem;}
.prob .n{white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:var(--muted);}
.prob .n.top{color:var(--text); font-weight:700;}
.prob .t{background:#1E2740; height:.6rem; border-radius:99px; overflow:hidden;}
.prob .f{height:100%; border-radius:99px;}
.prob .p{text-align:right; font:600 .82rem 'JetBrains Mono',monospace; color:var(--muted);}
.chip{display:inline-block; background:var(--surface2); border:1px solid var(--line); border-radius:8px;
      padding:.2rem .6rem; margin:.2rem .3rem .2rem 0; font:600 .8rem 'JetBrains Mono',monospace; color:var(--text);}
.chip i{color:var(--accent); font-style:normal; margin-left:.35rem;}
.evidence{background:var(--surface2); border:1px solid var(--line); border-radius:10px; padding:.9rem 1rem;
          line-height:1.7; font-size:.95rem; color:#C9D1E3;}
.evidence mark{background:rgba(255,176,32,.25); color:#FFD27A; border-radius:4px; padding:0 .2rem;}
.empty{border:1.5px dashed var(--line); border-radius:16px; padding:3.2rem 1.5rem; text-align:center; color:var(--muted);}
.empty b{display:block; color:var(--text); font:700 1.2rem 'Fraunces',serif; margin:.4rem 0;}

/* ---------- About */
.step{background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:1.1rem; height:100%;}
.step .num{font:700 .72rem 'JetBrains Mono',monospace; color:var(--accent); letter-spacing:.12em;}
.step h5{margin:.3rem 0 .4rem 0; font-size:1.15rem;}
.step p{color:var(--muted); font-size:.88rem; margin:0; line-height:1.5;}
.tax{background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:.9rem 1rem; margin-bottom:.7rem;
     border-left:4px solid var(--c);}
.tax b{font-size:1rem;} .tax div{color:var(--muted); font-size:.85rem; margin-top:.2rem;}
.dot{display:inline-block; width:.6rem; height:.6rem; border-radius:50%; margin-right:.5rem;}
.footer{display:flex; justify-content:space-between; flex-wrap:wrap; gap:1rem; color:var(--muted); font-size:.85rem;
        border-top:1px solid var(--line); margin-top:3rem; padding-top:1.3rem;}

[data-testid="stSidebar"]{background:var(--surface); border-right:1px solid var(--line);}
[data-testid="stFileUploader"] section{background:var(--surface2); border:1.5px dashed var(--line); border-radius:12px;}
@media (max-width:760px){.hero-title{font-size:2.2rem;} .prob{grid-template-columns:6.5rem 1fr 3rem;}}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================ Metadata
CATEGORY_INFO = {
    "BUSINESS": {"icon": "💼", "color": "#60A5FA", "desc": "Markets, finance, economy, companies and trade"},
    "ENTERTAINMENT": {"icon": "🎬", "color": "#A78BFA", "desc": "Cinema, television, music, celebrities and pop culture"},
    "FOOD & DRINK": {"icon": "🍲", "color": "#FB923C", "desc": "Culinary arts, recipes, restaurants and gastronomy"},
    "HEALTH & WELLNESS": {"icon": "🌿", "color": "#34D399", "desc": "Medicine, fitness, nutrition and mental health"},
    "PARENTING": {"icon": "👶", "color": "#F472B6", "desc": "Childcare, family life and parenting advice"},
    "POLITICS": {"icon": "🏛️", "color": "#F87171", "desc": "Government, elections, legislation and policy"},
    "QUEER VOICES": {"icon": "🌈", "color": "#2DD4BF", "desc": "LGBTQ+ community, identity, rights and culture"},
    "STYLE & BEAUTY": {"icon": "👗", "color": "#FB7185", "desc": "Fashion, design, cosmetics and lifestyle"},
    "TRAVEL": {"icon": "✈️", "color": "#38BDF8", "desc": "Tourism, destinations, flights, hotels and guides"},
    "WORLD NEWS": {"icon": "🌐", "color": "#94A3B8", "desc": "International affairs, conflicts and diplomacy"},
}
FALLBACK_META = {"icon": "📰", "color": "#94A3B8", "desc": ""}

SAMPLE_ARTICLES = {
    "🏛️ Politics": ("Senate passes bipartisan funding measure ahead of federal deadline",
        "Lawmakers in the Senate voted late Thursday to approve a comprehensive bipartisan spending package, sending the bill directly to the president's desk hours before government funding expired."),
    "💼 Business": ("Wall Street rallies as quarterly corporate earnings exceed analyst expectations",
        "Major stock market indices surged in heavy trading following robust quarterly earnings reports from leading technology and financial corporations."),
    "🎬 Entertainment": ("Director's new sci-fi epic dominates global box office with record debut",
        "The highly anticipated thriller secured the highest-grossing opening of the year. Critics praised the ensemble cast, visual effects, and original score."),
    "🍲 Food & Drink": ("A chef's seasonal guide to roasting wild mushrooms with herbs and garlic",
        "Chanterelles and king oyster mushrooms turn into a caramelized dinner centerpiece, sautéed in clarified butter with rosemary, thyme and lemon zest."),
    "🌿 Health": ("Daily brisk walking significantly reduces cardiovascular risk, clinical study finds",
        "Just 20 to 30 minutes of moderate aerobic walking per day lowers hypertension and heart disease risk, researchers found."),
    "✈️ Travel": ("Secret coastal villages in Southern Europe for a tranquil summer holiday",
        "Far from crowded resorts, secluded Mediterranean harbors offer pebble beaches, family-run tavernas and boutique stays at modest prices."),
    "👶 Parenting": ("Child psychologists share five bedtime routines that ease toddler sleep anxiety",
        "Predictable nighttime rituals help young children settle: dim screens an hour before bed, keep a warm bath routine, and read together."),
    "👗 Style": ("Spring runway collections champion vintage tailoring and circular fabrics",
        "Designers unveiled relaxed linen suits, oversized trench coats and zero-waste recycled textiles in warm terracotta and sage tones."),
    "🌐 World": ("International climate delegation signs landmark marine biodiversity treaty",
        "Envoys from over eighty nations ratified a treaty in Geneva establishing legally binding protections for international ocean corridors."),
}


# ============================================================ Backend
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        st.error(f"Model artifacts not found in `{RESULTS_DIR}`. Expected `best_news_model.joblib` and `model_metadata.joblib`.")
        st.stop()
    return joblib.load(MODEL_PATH), joblib.load(METADATA_PATH)


@st.cache_data
def load_comparison():
    return pd.read_csv(COMPARISON_CSV) if COMPARISON_CSV.exists() else None


model, metadata = load_artifacts()
comparison_df = load_comparison()
label_encoder = metadata["label_encoder"]
categories = list(label_encoder.classes_)
best_model_name = metadata.get("best_model", "Logistic Regression")
classifier = model.named_steps["classifier"]

best_row = None
if comparison_df is not None and not comparison_df.empty:
    best_row = comparison_df.loc[comparison_df["F1 Score (%)"].idxmax()]


def meta(cat: str) -> dict:
    return CATEGORY_INFO.get(cat, FALLBACK_META)


def predict_probs(texts) -> np.ndarray:
    """Class probabilities for a list of cleaned texts, with a softmax fallback."""
    if hasattr(classifier, "predict_proba"):
        return model.predict_proba(texts)
    if hasattr(classifier, "decision_function"):
        s = np.atleast_2d(model.decision_function(texts))
        e = np.exp(s - s.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    return np.full((len(texts), len(categories)), 1 / len(categories))


@st.cache_data(show_spinner=False)
def classify_text(text: str):
    cleaned = clean_text(text)
    if not cleaned or len(cleaned.split()) < 3:
        return None

    probs = predict_probs([cleaned])[0]
    order = np.argsort(probs)[::-1]
    ranked = [(categories[i], float(probs[i])) for i in order]
    pred_idx = int(order[0])

    top_terms = []
    try:
        tfidf = model.named_steps["tfidf"]
        names = np.array(tfidf.get_feature_names_out())
        vec = tfidf.transform([cleaned]).toarray()[0]
        nz = np.where(vec > 0)[0]
        if len(nz) and hasattr(classifier, "coef_"):
            w = vec[nz] * classifier.coef_[pred_idx, nz]
            for j in np.argsort(w)[::-1][:8]:
                if w[j] > 0.05:
                    top_terms.append((str(names[nz[j]]), round(float(w[j]), 2)))
    except Exception:
        pass

    return {"ranked": ranked, "top_terms": top_terms}


def highlight(text: str, terms) -> str:
    words = {w for t, _ in terms for w in t.split()}
    parts = re.split(r"(\b[\w']+\b)", text[:700])
    out = [f"<mark>{html.escape(p)}</mark>" if p.lower() in words else html.escape(p) for p in parts]
    return "".join(out) + ("…" if len(text) > 700 else "")


def kpi(label, value, sub="", color=None):
    style = f' style="color:{color}"' if color else ""
    return h(f'<div class="kpi"><div class="l">{label}</div><div class="v"{style}>{value}</div><div class="s">{sub}</div></div>')


# ============================================================ Sidebar
with st.sidebar:
    st.markdown("### 🗞️ Story Sorter")
    st.caption("Multiclass news classification with TF-IDF + scikit-learn.")
    st.markdown("---")
    st.markdown(f"**Active model**  \n{best_model_name}")
    if best_row is not None:
        st.caption(f"F1 {best_row['F1 Score (%)']:.2f}%  ·  Accuracy {best_row['Accuracy (%)']:.2f}%")
    st.markdown("---")
    st.markdown("**Categories**")
    st.markdown(
        h("".join(f'<div style="margin:.3rem 0;"><span class="dot" style="background:{meta(c)["color"]}"></span>{c.title()}</div>' for c in sorted(categories))),
        unsafe_allow_html=True,
    )

# ============================================================ Hero
stats_html = ""
if best_row is not None:
    stats_html = h(
        f"""<div class="hero-stats">
        <div class="stat"><b>{best_row['Accuracy (%)']:.1f}%</b><span>Accuracy</span></div>
        <div class="stat"><b>{best_row['F1 Score (%)']:.1f}%</b><span>Weighted F1</span></div>
        <div class="stat"><b>{len(categories)}</b><span>Desks</span></div></div>"""
    )

st.markdown(
    h(
        f"""<div class="hero"><div>
        <div class="eyebrow">NLP Lab · Story Sorter</div>
        <div class="hero-title">Every story, sent to the <em>right desk.</em></div>
        <p class="hero-sub">Paste a headline or article and get the predicted section, a confidence score, and the exact words that drove the decision.</p>
        </div>{stats_html}</div>"""
    ),
    unsafe_allow_html=True,
)

tab_live, tab_batch, tab_perf, tab_about = st.tabs(["Live classifier", "Batch CSV", "Model benchmarks", "Methodology"])

# ============================================================ TAB 1 — Live
with tab_live:
    for k in ("headline", "body"):
        st.session_state.setdefault(k, "")
    st.session_state.setdefault("last", None)

    def load_sample(key):
        title, body = SAMPLE_ARTICLES[key]
        st.session_state.headline, st.session_state.body = title, body
        st.session_state.autorun = True

    def clear_input():
        st.session_state.headline = ""
        st.session_state.body = ""
        st.session_state.last = None

    col_in, col_out = st.columns([1.05, 1], gap="large")

    with col_in:
        st.markdown('<div class="section-label">Try a sample</div>', unsafe_allow_html=True)
        keys = list(SAMPLE_ARTICLES)
        for row in range(0, len(keys), 3):
            cols = st.columns(3)
            for c, key in zip(cols, keys[row:row + 3]):
                c.button(key, on_click=load_sample, args=(key,), use_container_width=True, key=f"s_{key}")

        st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)
        with st.form("classify_form", border=True):
            st.text_input("Headline", key="headline", placeholder="Federal Reserve signals steady rate stance…")
            st.text_area("Article text", key="body", height=190, placeholder="Paste the lead paragraph or full story…")
            n_words = len(f"{st.session_state.headline} {st.session_state.body}".split())
            st.caption(f"{n_words} words · the model needs at least 3")
            b1, b2 = st.columns([2, 1])
            run = b1.form_submit_button("Classify story", type="primary", use_container_width=True)
            b2.form_submit_button("Clear", on_click=clear_input, use_container_width=True)

    # Run on submit or after a sample click
    autorun = st.session_state.pop("autorun", False)
    if run or autorun:
        text = f"{st.session_state.headline} {st.session_state.body}".strip()
        res = classify_text(text) if text else None
        st.session_state.last = {"text": text, "res": res}

    with col_out:
        st.markdown('<div class="section-label">Prediction</div>', unsafe_allow_html=True)
        last = st.session_state.last

        if last is None:
            st.markdown(
                h('<div class="empty">🗞️<b>Nothing classified yet</b>Pick a sample or paste a story, then hit Classify.</div>'),
                unsafe_allow_html=True,
            )
        elif last["res"] is None:
            st.markdown(
                h('<div class="empty">✍️<b>Not enough text</b>Enter at least three meaningful words.</div>'),
                unsafe_allow_html=True,
            )
        else:
            ranked, terms = last["res"]["ranked"], last["res"]["top_terms"]
            cat, conf = ranked[0]
            m = meta(cat)
            conf_pct = conf * 100
            level = "High" if conf >= 0.7 else "Medium" if conf >= 0.45 else "Low"

            bars = "".join(
                f'<div class="prob"><div class="n{" top" if i == 0 else ""}">{html.escape(n.title())}</div>'
                f'<div class="t"><div class="f" style="width:{p * 100:.1f}%; background:{meta(n)["color"] if i == 0 else "#4B5878"}"></div></div>'
                f'<div class="p">{p * 100:.1f}%</div></div>'
                for i, (n, p) in enumerate(ranked)
            )
            close = ""
            if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 0.08:
                close = f'<div class="callout"><b>Close call.</b> {html.escape(ranked[1][0].title())} is only {(ranked[0][1] - ranked[1][1]) * 100:.1f} points behind.</div>'

            chips = "".join(f'<span class="chip">{html.escape(t)}<i>+{w:.2f}</i></span>' for t, w in terms)
            terms_block = (
                f'<div class="section-label" style="margin-top:1.3rem">Words that drove it</div>{chips}'
                f'<div class="section-label" style="margin-top:1rem">Evidence in your text</div>'
                f'<div class="evidence">{highlight(last["text"], terms)}</div>'
            ) if terms else ""

            st.markdown(
                h(
                    f"""<div class="result">
                    <div class="result-top"><div>
                      <span class="badge">{level} confidence</span>
                      <div class="cat-name">{m['icon']} {html.escape(cat.title())}</div>
                      <p class="cat-desc">{m['desc']}</p></div>
                      <div class="ring" style="--p:{conf_pct:.0f}; --c:{m['color']}"><div>{conf_pct:.0f}%</div></div>
                    </div>{close}
                    {terms_block}
                    <div class="section-label" style="margin-top:1.3rem">All categories</div>{bars}
                    </div>"""
                ),
                unsafe_allow_html=True,
            )

# ============================================================ TAB 2 — Batch
with tab_batch:
    st.markdown("#### Classify a CSV in bulk")
    st.caption("Upload any CSV with headline or description columns. Results persist while you filter and explore.")

    c_up, c_tpl = st.columns([2, 1], gap="large")
    with c_up:
        uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    with c_tpl:
        tpl = pd.DataFrame([{"headline": t, "description": b} for t, b in SAMPLE_ARTICLES.values()])
        st.download_button("Download sample CSV", tpl.to_csv(index=False).encode(), "sample_news_batch.csv", "text/csv", use_container_width=True)
        st.caption("Tip: select several columns and they're merged into one text per row.")

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = None

        if df is not None and not df.empty:
            cols = df.columns.tolist()
            defaults = [c for c in cols if any(k in c.lower() for k in ("text", "title", "headline", "desc", "content"))] or cols[:1]
            selected = st.multiselect("Text columns", cols, default=defaults)

            if st.button("Run batch prediction", type="primary"):
                if not selected:
                    st.error("Select at least one column.")
                else:
                    with st.spinner(f"Classifying {len(df):,} rows…"):
                        combined = df[selected].fillna("").astype(str).agg(" ".join, axis=1)
                        cleaned = combined.apply(clean_text).tolist()
                        probs = predict_probs(cleaned)
                        out = df.copy()
                        out["predicted_category"] = [categories[i] for i in probs.argmax(axis=1)]
                        out["confidence"] = (probs.max(axis=1) * 100).round(1)
                        out.loc[[len(t.split()) < 3 for t in cleaned], ["predicted_category", "confidence"]] = [None, None]
                    st.session_state.batch = {"out": out, "name": uploaded.name}

    if st.session_state.get("batch"):
        out = st.session_state.batch["out"]
        ok = out.dropna(subset=["predicted_category"])
        st.markdown("---")
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(kpi("Rows", f"{len(out):,}", f"{len(out) - len(ok)} too short to classify" if len(ok) < len(out) else "all classified"), unsafe_allow_html=True)
        top = ok["predicted_category"].mode()[0] if len(ok) else "—"
        k2.markdown(kpi("Top category", top.title() if top != "—" else top, f"{(ok['predicted_category'] == top).mean() * 100:.0f}% of rows" if len(ok) else ""), unsafe_allow_html=True)
        k3.markdown(kpi("Mean confidence", f"{ok['confidence'].mean():.1f}%" if len(ok) else "—"), unsafe_allow_html=True)
        k4.markdown(kpi("Low confidence", f"{(ok['confidence'] < 45).sum():,}", "rows under 45%", "#FBBF24"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        cc, ct = st.columns([1, 1.4], gap="large")
        with cc:
            counts = ok["predicted_category"].value_counts().rename_axis("category").reset_index(name="rows")
            chart = (
                alt.Chart(counts)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    y=alt.Y("category:N", sort="-x", title=None),
                    x=alt.X("rows:Q", title=None),
                    color=alt.Color("category:N", legend=None,
                                    scale=alt.Scale(domain=list(CATEGORY_INFO), range=[v["color"] for v in CATEGORY_INFO.values()])),
                    tooltip=["category", "rows"],
                )
                .properties(height=max(220, 28 * len(counts)))
                .configure_view(strokeOpacity=0)
                .configure_axis(labelColor="#8B96AE", gridColor="#1E2740", domainColor="#26304A")
            )
            st.altair_chart(chart, use_container_width=True)
        with ct:
            pick = st.selectbox("Filter by category", ["All"] + sorted(ok["predicted_category"].unique().tolist()))
            view = out if pick == "All" else out[out["predicted_category"] == pick]
            st.dataframe(
                view,
                use_container_width=True,
                height=340,
                column_config={"confidence": st.column_config.ProgressColumn("confidence", min_value=0, max_value=100, format="%.1f%%")},
            )
        st.download_button(
            "Download classified CSV",
            out.to_csv(index=False).encode(),
            f"classified_{st.session_state.batch['name']}",
            "text/csv",
            type="primary",
        )

# ============================================================ TAB 3 — Benchmarks
with tab_perf:
    st.markdown("#### Model benchmarks")
    st.caption("Supervised models compared on the held-out test split.")

    if comparison_df is None:
        st.info("`results/model_comparison.csv` not found.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(kpi("Champion", best_row["Model"], "highest weighted F1", "#FFB020"), unsafe_allow_html=True)
        k2.markdown(kpi("Weighted F1", f"{best_row['F1 Score (%)']:.2f}%", f"across {len(categories)} classes", "#34D399"), unsafe_allow_html=True)
        k3.markdown(kpi("Accuracy", f"{best_row['Accuracy (%)']:.2f}%", "held-out test set"), unsafe_allow_html=True)
        k4.markdown(kpi("Training time", f"{best_row['Training Time (seconds)']:.1f}s", "this model"), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        metric_cols = [c for c in ("Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)") if c in comparison_df]
        left, right = st.columns([1, 1.15], gap="large")
        with left:
            st.markdown("##### Leaderboard")
            st.dataframe(
                comparison_df.sort_values("F1 Score (%)", ascending=False).reset_index(drop=True),
                use_container_width=True,
                column_config={c: st.column_config.ProgressColumn(c, min_value=0, max_value=100, format="%.2f") for c in metric_cols},
            )
        with right:
            st.markdown("##### Metric comparison")
            long = comparison_df.melt(id_vars="Model", value_vars=metric_cols, var_name="Metric", value_name="Score")
            ymin = max(0, float(long["Score"].min()) - 5)
            chart = (
                alt.Chart(long)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("Model:N", title=None, axis=alt.Axis(labelAngle=-20)),
                    xOffset="Metric:N",
                    y=alt.Y("Score:Q", scale=alt.Scale(domain=[ymin, 100]), title=None),
                    color=alt.Color("Metric:N", scale=alt.Scale(range=["#5B8CFF", "#A78BFA", "#34D399", "#FFB020"]),
                                    legend=alt.Legend(orient="top", title=None, labelColor="#C9D1E3")),
                    tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".2f")],
                )
                .properties(height=330)
                .configure_view(strokeOpacity=0)
                .configure_axis(labelColor="#8B96AE", gridColor="#1E2740", domainColor="#26304A")
            )
            st.altair_chart(chart, use_container_width=True)

        st.markdown("---")
        st.markdown("##### Confusion matrices")
        candidates = {
            "Logistic Regression": "logistic_regression_confusion_matrix.png",
            "Linear SVM": "linear_svm_confusion_matrix.png",
            "SVM (RBF)": "svm_rbf_confusion_matrix.png",
            "Naive Bayes": "naive_bayes_confusion_matrix.png",
            "XGBoost": "xgboost_confusion_matrix.png",
            "Full comparison plot": "model_comparison.png",
        }
        available = {k: RESULTS_DIR / v for k, v in candidates.items() if (RESULTS_DIR / v).exists()}
        if available:
            choice = st.radio("Model", list(available), horizontal=True, label_visibility="collapsed")
            st.image(str(available[choice]), caption=choice, use_container_width=True)
        else:
            st.caption("No confusion-matrix images found in `results/`.")

# ============================================================ TAB 4 — Methodology
with tab_about:
    st.markdown("#### How it works")
    s1, s2, s3 = st.columns(3, gap="medium")
    steps = [
        ("01 · Clean", "Preprocessing", "Lowercase, strip URLs and HTML, drop non-alphabetic characters, collapse whitespace."),
        ("02 · Vectorize", "TF-IDF", "Unigrams + bigrams, English stop words removed, sublinear term frequency, 20,000-feature vocabulary."),
        ("03 · Classify", best_model_name, "A linear classifier scores each category; probabilities are shown for all of them."),
    ]
    for col, (num, title, body) in zip((s1, s2, s3), steps):
        col.markdown(h(f'<div class="step"><div class="num">{num}</div><h5>{title}</h5><p>{body}</p></div>'), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Category taxonomy")
    cols = st.columns(2, gap="medium")
    for i, (name, info) in enumerate(CATEGORY_INFO.items()):
        cols[i % 2].markdown(
            h(f'<div class="tax" style="--c:{info["color"]}"><b>{info["icon"]} {name.title()}</b><div>{info["desc"]}</div></div>'),
            unsafe_allow_html=True,
        )

    st.markdown("#### Good to know")
    st.markdown(
        "- **Trained on HuffPost headlines and short descriptions.** Long articles or other styles may be less accurate.\n"
        "- **Single-label output.** Stories that span two desks (e.g. political satire) are forced into one; watch for the *close call* warning.\n"
        "- **Term weights are a linear approximation**, so treat highlighted words as hints about the model, not proof of meaning."
    )

# ============================================================ Footer
st.markdown(
    h('<div class="footer"><div>Story Sorter · Applied Machine Learning Lab</div>'
      f'<div>Trained on the Kaggle News Category dataset · {len(categories)} categories</div></div>'),
    unsafe_allow_html=True,
)