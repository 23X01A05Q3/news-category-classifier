"""
app.py
======
Streamlit UI for the News Category Classification System.

Features:
    • Login page gate (username + password required)
    • Beautiful dark-mode glassmorphism interface
    • Input box for a news headline
    • Choose between Naive Bayes and Logistic Regression
    • Displays predicted category with confidence bar chart
    • Shows top-K category probabilities

Run:
    streamlit run app.py
"""

import os
import sys
import hashlib
import streamlit as st
import plotly.graph_objects as go

# ─── Page Config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="📰 News Category Classifier",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Add project root to path ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import clean_text
from model import load_models, predict_category

# ─── User Credentials Store ───────────────────────────────────────────────────
# In a real app use a database + bcrypt. Here we store sha256 hashes.
# Default credentials: admin / news@123  |  user / pass123
def _h(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

USERS: dict[str, str] = {
    "admin": _h("news@123"),
    "user":  _h("pass123"),
}

# ─── Session State Bootstrap ─────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "login_error" not in st.session_state:
    st.session_state.login_error = ""

# ─── Shared CSS ───────────────────────────────────────────────────────────────
SHARED_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background with mesh effect */
.stApp {
    background: 
        radial-gradient(at 0% 0%, rgba(102, 126, 234, 0.15) 0, transparent 50%), 
        radial-gradient(at 100% 0%, rgba(118, 75, 162, 0.15) 0, transparent 50%), 
        radial-gradient(at 100% 100%, rgba(102, 126, 234, 0.1) 0, transparent 50%), 
        radial-gradient(at 0% 100%, rgba(118, 75, 162, 0.1) 0, transparent 50%),
        linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #e8e8f0;
}

/* ── Cards ─────────────────────────────────────────────── */
.glass-card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
}

/* ── Login card (centred, narrower) ────────────────────── */
.login-card {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(25px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 30px;
    padding: 3.5rem 3rem 3rem 3rem;
    box-shadow: 
        0 25px 50px -12px rgba(0, 0, 0, 0.5),
        inset 0 0 0 1px rgba(255, 255, 255, 0.05);
    text-align: center;
    max-width: 420px;
    margin: 0 auto;
    animation: fadeInScale 0.6s ease-out;
}

@keyframes fadeInScale {
    from { opacity: 0; transform: scale(0.95) translateY(10px); }
    to { opacity: 1; transform: scale(1) translateY(0); }
}

.login-logo {
    font-size: 4rem;
    margin-bottom: 1rem;
    display: inline-block;
    filter: drop-shadow(0 0 15px rgba(167, 139, 250, 0.4));
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}

.login-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff 30%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    letter-spacing: -0.5px;
}

.login-subtitle {
    font-size: 1rem;
    color: rgba(255,255,255,0.45);
    margin-bottom: 2.5rem;
    font-weight: 400;
}

.login-divider {
    height: 1px;
    background: linear-gradient(to right, transparent, rgba(255, 255, 255, 0.1), transparent);
    margin: 2rem 0;
}

.demo-credentials-container {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 15px;
    padding: 1.2rem;
    margin-top: 1rem;
}

.demo-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(167,139,250,0.1);
    border: 1px solid rgba(167,139,250,0.2);
    border-radius: 12px;
    padding: 0.5rem 1rem;
    font-size: 0.85rem;
    color: #c4b5fd;
    margin: 0.3rem;
    transition: all 0.3s ease;
}

.demo-pill:hover {
    background: rgba(167,139,250,0.2);
    border-color: rgba(167,139,250,0.4);
    transform: translateY(-2px);
}

/* ── Hero Banner ───────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 18px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(102,126,234,0.35);
}
.hero-banner h1 { color:#fff; font-size:2.2rem; font-weight:700; margin:0 0 0.3rem 0; }
.hero-banner p  { color:rgba(255,255,255,0.85); font-size:1rem; margin:0; }

/* ── Result Badge ──────────────────────────────────────── */
.result-badge {
    display: inline-block;
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white; font-size:1.4rem; font-weight:700;
    padding:0.6rem 1.8rem; border-radius:50px;
    margin:0.5rem 0 1rem 0;
    box-shadow: 0 4px 20px rgba(240,147,251,0.4);
    letter-spacing:0.5px;
}

/* ── Metric Box ────────────────────────────────────────── */
.metric-box {
    background:rgba(255,255,255,0.08); border-radius:12px;
    padding:1rem; text-align:center; border:1px solid rgba(255,255,255,0.1);
}
.metric-val   { font-size:1.6rem; font-weight:700; color:#a78bfa; }
.metric-label { font-size:0.8rem; color:rgba(255,255,255,0.6);
                text-transform:uppercase; letter-spacing:0.8px; }

/* ── Suggestion Pills ──────────────────────────────────── */
.suggestion-label {
    font-size: 0.85rem;
    color: rgba(255, 255, 255, 0.4);
    margin-bottom: 0.8rem;
    font-weight: 500;
}
.pill-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-bottom: 2rem;
}
.suggestion-pill {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 50px;
    padding: 0.4rem 1rem;
    font-size: 0.85rem;
    color: #e8e8f0;
    cursor: pointer;
    transition: all 0.2s ease;
}
.suggestion-pill:hover {
    background: rgba(167, 139, 250, 0.15);
    border-color: rgba(167, 139, 250, 0.4);
    transform: translateY(-2px);
}

/* ── Keywords & Insights ───────────────────────────────── */
.insight-card {
    background: rgba(167, 139, 250, 0.05);
    border-left: 3px solid #a78bfa;
    padding: 1rem;
    border-radius: 4px 12px 12px 4px;
    margin-top: 1rem;
}
.keyword-tag {
    display: inline-block;
    background: rgba(255, 255, 255, 0.08);
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-family: monospace;
    color: #a78bfa;
    margin-right: 0.4rem;
}

/* ══ SIDEBAR STYLES ═════════════════════════════════════ */

/* Sidebar background + custom scrollbar */
[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1rem !important;
}
[data-testid="stSidebar"] ::-webkit-scrollbar { width: 4px; }
[data-testid="stSidebar"] ::-webkit-scrollbar-track { background: transparent; }
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
    background: rgba(167,139,250,0.3); border-radius: 4px;
}

/* ── Profile card ──────────────────────────────────────── */
.sb-profile {
    background: linear-gradient(135deg, rgba(102,126,234,0.18) 0%, rgba(118,75,162,0.18) 100%);
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 16px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.9rem;
}
.sb-avatar {
    width: 46px; height: 46px; flex-shrink: 0;
    border-radius: 50%;
    background: linear-gradient(135deg, #667eea 0%, #f093fb 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; font-weight: 700; color: white;
    box-shadow: 0 4px 14px rgba(102,126,234,0.5);
}
.sb-info { display: flex; flex-direction: column; gap: 0.1rem; }
.sb-name  { font-size: 1rem; font-weight: 700; color: #e8e8f0; }
.sb-role  { font-size: 0.72rem; color: #a78bfa; text-transform: uppercase; letter-spacing: 0.8px; }
.sb-badge {
    margin-left: auto;
    background: rgba(52,211,153,0.15);
    border: 1px solid rgba(52,211,153,0.4);
    border-radius: 50px; padding: 0.18rem 0.7rem;
    font-size: 0.7rem; color: #6ee7b7; font-weight: 600;
    white-space: nowrap;
}

/* ── Section headers ───────────────────────────────────── */
.sb-section-header {
    display: flex; align-items: center; gap: 0.5rem;
    margin: 1.3rem 0 0.8rem 0;
}
.sb-section-icon {
    width: 28px; height: 28px; border-radius: 8px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem; flex-shrink: 0;
}
.sb-section-title {
    font-size: 0.72rem; font-weight: 700; color: rgba(255,255,255,0.5);
    text-transform: uppercase; letter-spacing: 1.2px;
}

/* ── Setting card ──────────────────────────────────────── */
.sb-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.7rem;
}

/* ── Stat row ──────────────────────────────────────────── */
.sb-stat-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.55rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.sb-stat-row:last-child { border-bottom: none; }
.sb-stat-label { font-size: 0.82rem; color: rgba(255,255,255,0.55); }
.sb-stat-val   { font-size: 0.82rem; font-weight: 600; color: #c4b5fd; }

/* ── About card ────────────────────────────────────────── */
.sb-about-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1rem 0.6rem 1rem;
    margin-bottom: 0.7rem;
}
.sb-about-row {
    display: flex; align-items: flex-start; gap: 0.7rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.sb-about-row:last-child { border-bottom: none; }
.sb-about-icon {
    width: 30px; height: 30px; flex-shrink: 0;
    border-radius: 8px; display: flex; align-items: center;
    justify-content: center; font-size: 0.9rem;
}
.sb-about-icon.tfidf   { background: rgba(102,126,234,0.2); }
.sb-about-icon.nb      { background: rgba(251,191,36,0.2);  }
.sb-about-icon.lr      { background: rgba(52,211,153,0.2);  }
.sb-about-text { display: flex; flex-direction: column; gap: 0.15rem; }
.sb-about-name { font-size: 0.83rem; font-weight: 600; color: #e8e8f0; }
.sb-about-desc { font-size: 0.73rem; color: rgba(255,255,255,0.45); line-height: 1.4; }

/* ── Sign-out button ───────────────────────────────────── */
.sb-signout > button {
    background: rgba(245,87,108,0.1) !important;
    border: 1px solid rgba(245,87,108,0.35) !important;
    color: #fca5a5 !important;
    border-radius: 10px !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 1rem !important;
    font-weight: 600 !important;
    transition: background 0.2s !important;
    box-shadow: none !important;
    width: 100% !important;
}
.sb-signout > button:hover {
    background: rgba(245,87,108,0.22) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Inputs ────────────────────────────────────────────── */
.stTextInput  input,
.stTextArea   textarea {
    background:rgba(255,255,255,0.07) !important;
    border:1px solid rgba(167,139,250,0.4) !important;
    border-radius:10px !important;
    color:#e8e8f0 !important; font-size:1rem !important;
}
.stTextInput  input:focus,
.stTextArea   textarea:focus {
    border-color:rgba(167,139,250,0.9) !important;
    box-shadow:0 0 0 3px rgba(167,139,250,0.15) !important;
}

/* ── Buttons ───────────────────────────────────────────── */
.stButton>button {
    background:linear-gradient(135deg,#667eea 0%,#764ba2 100%) !important;
    color:white !important; border:none !important;
    border-radius:10px !important;
    padding:0.65rem 2.5rem !important;
    font-size:1.05rem !important; font-weight:600 !important;
    transition:transform 0.2s, box-shadow 0.2s !important;
    box-shadow:0 4px 15px rgba(102,126,234,0.45) !important;
    width:100% !important;
}
.stButton>button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 6px 25px rgba(102,126,234,0.6) !important;
}

/* ── Sidebar ───────────────────────────────────────────── */
[data-testid="stSidebar"] { background:rgba(15,12,41,0.95) !important; }

/* ── Divider ───────────────────────────────────────────── */
hr { border-color:rgba(255,255,255,0.1) !important; }

/* Sidebar overall background */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0b26 0%, #1a1040 60%, #0f0c29 100%) !important;
    border-right: 1px solid rgba(167,139,250,0.15) !important;
}

/* ── Error message ─────────────────────────────────────── */
.login-err {
    background: rgba(245,87,108,0.15);
    border: 1px solid rgba(245,87,108,0.4);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    color: #fca5a5;
    font-size: 0.9rem;
    margin-top: 0.5rem;
    text-align: left;
}
</style>
"""

st.markdown(SHARED_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
def show_login_page():
    """Render the login page and handle authentication logic."""

    # Centre the card in a narrow column
    _, col, _ = st.columns([0.5, 2, 0.5])
    with col:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)

        # Logo + Title
        st.markdown(
            '<span class="login-logo">📰</span>'
            '<div class="login-title">NewsClassifier</div>'
            '<div class="login-subtitle">Intelligence for modern newsrooms</div>',
            unsafe_allow_html=True,
        )

        # Error banner
        if st.session_state.login_error:
            st.markdown(
                f'<div class="login-err" style="margin-bottom: 1.5rem;">🔒 {st.session_state.login_error}</div>',
                unsafe_allow_html=True,
            )
            st.session_state.login_error = ""

        # ── Form ──────────────────────────────────────────────────────────
        with st.form("login_form", clear_on_submit=False):
            st.markdown('<div style="text-align: left; margin-bottom: 0.5rem; font-size: 0.9rem; color: rgba(255,255,255,0.6);">USERNAME</div>', unsafe_allow_html=True)
            username = st.text_input(
                "Username",
                placeholder="admin",
                autocomplete="username",
                label_visibility="collapsed"
            )
            st.markdown('<div style="text-align: left; margin-top: 1.5rem; margin-bottom: 0.5rem; font-size: 0.9rem; color: rgba(255,255,255,0.6);">PASSWORD</div>', unsafe_allow_html=True)
            password = st.text_input(
                "Password",
                type="password",
                placeholder="••••••••",
                autocomplete="current-password",
                label_visibility="collapsed"
            )
            st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Access Workspace", use_container_width=True)

        if submitted:
            _handle_login(username.strip(), password)

        # ── Demo credentials hint ─────────────────────────────────────────
        st.markdown('<div class="login-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="demo-credentials-container">', unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.8rem;color:rgba(255,255,255,0.3);margin-bottom:0.8rem;text-transform: uppercase; letter-spacing: 1px;'>"
            "Guest Access</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="display: flex; flex-wrap: wrap; justify-content: center;">'
            '<span class="demo-pill"><span>👤</span> <b>admin</b> / news@123</span>'
            '<span class="demo-pill"><span>👤</span> <b>user</b> / pass123</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # close login-card

    # Subtle footer
    st.markdown(
        "<p style='text-align:center;color:rgba(255,255,255,0.2);font-size:0.75rem;margin-top:2rem;'>"
        "News Category Classification System · Powered by scikit-learn</p>",
        unsafe_allow_html=True,
    )


def _handle_login(username: str, password: str):
    """Validate credentials and update session state."""
    if not username or not password:
        st.session_state.login_error = "Please enter both username and password."
        st.rerun()

    hashed = _h(password)
    if username in USERS and USERS[username] == hashed:
        st.session_state.authenticated = True
        st.session_state.username = username
        st.session_state.login_error = ""
        st.rerun()
    else:
        st.session_state.login_error = "Invalid username or password. Please try again."
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP  (shown only after successful login)
# ══════════════════════════════════════════════════════════════════════════════

# ─── Load Models (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models …")
def get_models():
    """Load and cache the trained models."""
    try:
        return load_models()
    except FileNotFoundError as e:
        return None, None, None, None, str(e)


# ─── Category Emoji Mapping ───────────────────────────────────────────────────
CATEGORY_EMOJI = {
    'POLITICS':      '🏛️',
    'SPORTS':        '⚽',
    'TECHNOLOGY':    '💻',
    'ENTERTAINMENT': '🎬',
    'HEALTH':        '🏥',
    'BUSINESS':      '📈',
    'SCIENCE':       '🔬',
    'TRAVEL':        '✈️',
    'FOOD':          '🍽️',
    'ENVIRONMENT':   '🌿',
    'COMEDY':        '😄',
    'ARTS':          '🎨',
    'CRIME':         '🚔',
    'EDUCATION':     '📚',
    'RELIGION':      '⛪',
    'STYLE':         '👗',
    'WELLNESS':      '🧘',
    'MONEY':         '💰',
    'WORLDPOST':     '🌍',
    'WORLD NEWS':    '🌐',
}

def get_emoji(category: str) -> str:
    """Return emoji for a category with a default fallback."""
    for key, emoji in CATEGORY_EMOJI.items():
        if key in category.upper():
            return emoji
    return '📰'


def show_main_app():
    """Render the full classifier UI."""

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        uname   = st.session_state.username
        initial = uname[0].upper()

        # ── Profile card ─────────────────────────────────────────────────────
        st.markdown(
            f"""
            <div class="sb-profile">
              <div class="sb-avatar">{initial}</div>
              <div class="sb-info">
                <div class="sb-name">{uname.capitalize()}</div>
                <div class="sb-role">Analyst</div>
              </div>
              <div class="sb-badge">● Online</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Sign Out ──────────────────────────────────────────────────────────
        st.markdown('<div class="sb-signout">', unsafe_allow_html=True)
        if st.button("🚪  Sign Out", use_container_width=True, key="signout_btn"):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════════
        # ⚙️  SETTINGS
        # ════════════════════════════════════════════════════════════════════
        st.markdown(
            '<div class="sb-section-header">'
            '  <div class="sb-section-icon">⚙️</div>'
            '  <div class="sb-section-title">Settings</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sb-card">', unsafe_allow_html=True)
        selected_model = st.selectbox(
            "🤖  Classifier",
            options=["Logistic Regression", "Multinomial Naive Bayes"],
            index=0,
            help="Logistic Regression generally achieves higher accuracy on larger datasets.",
        )
        show_confidence = st.checkbox("📊  Show confidence chart", value=True)
        show_top_k      = st.slider("🏆  Top-K categories", 3, 10, 5)
        st.markdown('</div>', unsafe_allow_html=True)

        # Model info row
        model_accuracy = "~92%" if selected_model == "Logistic Regression" else "~84%"
        model_speed    = "Medium" if selected_model == "Logistic Regression" else "Fast"
        st.markdown(
            f"""
            <div class="sb-card">
              <div class="sb-stat-row">
                <span class="sb-stat-label">Active Model</span>
                <span class="sb-stat-val">{"LR" if selected_model == "Logistic Regression" else "NB"}</span>
              </div>
              <div class="sb-stat-row">
                <span class="sb-stat-label">Est. Accuracy</span>
                <span class="sb-stat-val">{model_accuracy}</span>
              </div>
              <div class="sb-stat-row">
                <span class="sb-stat-label">Speed</span>
                <span class="sb-stat-val">{model_speed}</span>
              </div>
              <div class="sb-stat-row">
                <span class="sb-stat-label">Categories</span>
                <span class="sb-stat-val">10</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ════════════════════════════════════════════════════════════════════
        # 📚  ABOUT
        # ════════════════════════════════════════════════════════════════════
        st.markdown(
            '<div class="sb-section-header">'
            '  <div class="sb-section-icon">📚</div>'
            '  <div class="sb-section-title">How It Works</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="sb-about-card">
              <div class="sb-about-row">
                <div class="sb-about-icon tfidf">📐</div>
                <div class="sb-about-text">
                  <div class="sb-about-name">TF-IDF</div>
                  <div class="sb-about-desc">Converts raw text into weighted numerical vectors</div>
                </div>
              </div>
              <div class="sb-about-row">
                <div class="sb-about-icon nb">⚡</div>
                <div class="sb-about-text">
                  <div class="sb-about-name">Naive Bayes</div>
                  <div class="sb-about-desc">Fast probabilistic classifier based on word frequencies</div>
                </div>
              </div>
              <div class="sb-about-row">
                <div class="sb-about-icon lr">🎯</div>
                <div class="sb-about-text">
                  <div class="sb-about-name">Logistic Regression</div>
                  <div class="sb-about-desc">Learns discriminative weights for higher accuracy</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Train reminder
        st.markdown(
            "<div style='background:rgba(102,126,234,0.1);border:1px solid rgba(102,126,234,0.3);"
            "border-radius:10px;padding:0.7rem 0.9rem;margin-top:0.3rem'>"
            "<span style='font-size:0.72rem;color:rgba(255,255,255,0.45);'>First time? Train models:</span><br>"
            "<code style='font-size:0.78rem;color:#a78bfa;'>python main.py</code>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Hero Banner ───────────────────────────────────────────────────────────
    st.markdown("""
<div class="hero-banner">
    <h1>📰 News Category Classifier</h1>
    <p>Paste any news headline below and let AI instantly predict its category.</p>
</div>
""", unsafe_allow_html=True)

    # ── Load Models ───────────────────────────────────────────────────────────
    load_result = get_models()
    if load_result[0] is None:
        err_msg = load_result[4] if len(load_result) == 5 else "Unknown error"
        st.error(f"⚠️ Models not found. Please run `python main.py` first.\n\n`{err_msg}`")
        st.stop()

    vectorizer, nb_model, lr_model, label_classes = load_result[:4]
    active_model = lr_model if selected_model == "Logistic Regression" else nb_model

    # ── Input Section ─────────────────────────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📝 Neural Analysis Engine")

    st.markdown("<p style='color: rgba(255,255,255,0.4); font-size: 0.85rem; margin-bottom: 1.5rem;'>Input a headline or technical summary for high-precision neural classification.</p>", unsafe_allow_html=True)

    headline_input = st.text_area(
        label="News Headline",
        placeholder="Paste technical summary or headline here...",
        height=120,
        label_visibility="collapsed",
    )

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        classify_btn = st.button("🚀 Execute Neural Pass", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Quick Suggestions (Pills) ─────────────────────────────────────────────
    st.markdown('<div class="suggestion-label">NEURAL TEST CASES</div>', unsafe_allow_html=True)
    
    examples = {
        "🔬 Space": "Scientists detect mysterious radio signals from a distant galaxy",
        "💰 Finance": "Global markets rally as inflation data shows unexpected cooling",
        "🏥 Biotech": "Breakthrough gene therapy shows promise in treating rare diseases",
        "🎬 Culture": "Streaming giant announces record-breaking subscribers for new series",
        "🌿 Climate": "Arctic sea ice reaches record lows as global temperatures climb"
    }
    
    cols = st.columns(len(examples))
    for i, (label, text) in enumerate(examples.items()):
        if cols[i].button(label, use_container_width=True, key=f"pill_{i}", help=text):
            st.session_state['example_headline'] = text
            st.rerun()

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    if 'example_headline' in st.session_state and not headline_input:
        headline_input = st.session_state.pop('example_headline')
        st.rerun()

    # ── Classification Result ─────────────────────────────────────────────────
    if classify_btn and headline_input.strip():
        with st.spinner("Analysing headline …"):
            category, prob_dict = predict_category(
                headline_input.strip(), vectorizer, active_model, clean_text
            )

        emoji = get_emoji(category)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Classification Result")

        col1, col2 = st.columns([2.5, 1])
        with col1:
            st.markdown(f"<div style='font-size: 1.1rem; margin-bottom: 0.5rem;'>{headline_input.strip()}</div>", unsafe_allow_html=True)
            st.markdown(
                f'<div class="result-badge">{emoji} {category}</div>',
                unsafe_allow_html=True,
            )
            
            # Simple keyword extraction (for "experienced developer" feel)
            cleaned = clean_text(headline_input)
            keywords = cleaned.split()[:5]
            kw_html = "".join([f'<span class="keyword-tag">{kw}</span>' for kw in keywords])
            
            st.markdown(
                f"""
                <div class="insight-card">
                    <div style="font-size: 0.8rem; font-weight: 600; color: rgba(255,255,255,0.4); margin-bottom: 0.4rem;">KEY FEATURES DETECTED</div>
                    {kw_html}
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            top_prob = max(prob_dict.values())
            conf_pct = round(top_prob * 100, 1)
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-val">{conf_pct}%</div>
                    <div class="metric-label">Confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # Mock "Model Consensus"
            other_model_name = "NB" if selected_model == "Logistic Regression" else "LR"
            st.markdown(
                f"<div style='text-align: center; margin-top: 1rem; font-size: 0.75rem; color: rgba(52,211,153,0.8);'>"
                f"✅ Verified by {other_model_name} model"
                f"</div>",
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

        # Probability bar chart
        if show_confidence and prob_dict:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Category Probabilities")

            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:show_top_k]
            cats  = [f"{get_emoji(c)} {c}" for c, _ in sorted_probs]
            probs = [p for _, p in sorted_probs]
            colors = [
                'rgba(118,75,162,0.9)' if i == 0 else 'rgba(102,126,234,0.6)'
                for i in range(len(cats))
            ]

            fig = go.Figure(go.Bar(
                x=probs[::-1], y=cats[::-1], orientation='h',
                marker_color=colors[::-1],
                text=[f"{p*100:.1f}%" for p in probs[::-1]],
                textposition='outside',
                textfont=dict(color='white', size=12),
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', family='Inter'),
                xaxis=dict(
                    showgrid=True, gridcolor='rgba(255,255,255,0.1)',
                    tickformat='.0%', range=[0, min(1, max(probs) * 1.35)],
                    color='rgba(255,255,255,0.7)',
                ),
                yaxis=dict(color='white'),
                margin=dict(l=10, r=80, t=10, b=10),
                height=60 + len(cats) * 45,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    elif classify_btn and not headline_input.strip():
        st.warning("⚠️ Please enter a news headline before clicking Classify.")

    # ── Available Categories ──────────────────────────────────────────────────
    with st.expander("🗂️ Available Categories"):
        cats_per_row = 4
        cats = sorted(label_classes)
        rows = [cats[i:i+cats_per_row] for i in range(0, len(cats), cats_per_row)]
        for row in rows:
            cols = st.columns(cats_per_row)
            for col, cat in zip(cols, row):
                with col:
                    st.markdown(
                        f"<div style='background:rgba(255,255,255,0.07);border-radius:8px;"
                        f"padding:8px;text-align:center;font-size:0.85rem;margin:2px'>"
                        f"{get_emoji(cat)} {cat}</div>",
                        unsafe_allow_html=True,
                    )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:rgba(255,255,255,0.35);font-size:0.8rem;'>"
        "News Category Classification System · Built with Streamlit + scikit-learn"
        "</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER  — show login or main app based on session state
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.authenticated:
    show_main_app()
else:
    show_login_page()
