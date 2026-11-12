import base64
import contextlib
import hashlib
import hmac
import json
import os
import random
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import sys
import time
from sqlalchemy import text

module_dir = "./"
sys.path.append(module_dir)

# ==========================================
# CONFIGURATION
# ==========================================

# Define both payload paths
PAYLOAD_FILE_RNA = "./label_evaluation/data/survey_data_RNA.json"
PAYLOAD_FILE_MA = "./label_evaluation/data/survey_data_MA.json"

# Unified results file for the combined evaluation
RESULTS_FILE = "./label_evaluation/data/evaluation_results_combined.csv"

# Motivation features
TEAM_GOALS = [500, 1000]                           # stepped team goals (unique samples reviewed by anyone)
MILESTONES = {10, 25, 50, 100, 250, 500, 1000}     # personal totals that trigger a celebration (+ every 500 after 1000)

st.set_page_config(layout="wide", page_title="Label Evaluation Tool")

# ==========================================
# DATABASE CONNECTION
# ==========================================

def get_db_connection():
    """Single shared DB connection. For Postgres, psycopg's automatic prepared statements are disabled
    (prepare_threshold=None): the Supabase transaction pooler (port 6543) hands each transaction to a
    different server connection, so a statement prepared on one is missing on the next ->
    InvalidSqlStatementName. pool_pre_ping also replaces connections the pooler has silently dropped."""
    if str(st.secrets["connections"]["results_db"].get("url", "")).startswith("postgresql"):
        return st.connection("results_db", type="sql",
                             connect_args={"prepare_threshold": None}, pool_pre_ping=True)
    return st.connection("results_db", type="sql")


# ==========================================
# USER MANAGEMENT (DB-backed, bcrypt)
# ==========================================

def _ensure_users_table(connection):
    """Create users table and seed legacy accounts on first run."""
    import bcrypt
    with connection.session as s:
        s.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        legacy = {"researcher1": "pass123", "researcher2": "tulip2026", "alex": "admin"}
        for uname, pwd in legacy.items():
            existing = s.execute(
                text("SELECT 1 FROM users WHERE username = :u"), {"u": uname}
            ).fetchone()
            if not existing:
                hashed = bcrypt.hashpw(pwd.encode(), bcrypt.gensalt()).decode()
                s.execute(
                    text("INSERT INTO users (username, password_hash) VALUES (:u, :h)"),
                    {"u": uname, "h": hashed}
                )
        s.commit()


def _verify_password(connection, username: str, password: str) -> bool:
    import bcrypt
    with connection.session as s:
        row = s.execute(
            text("SELECT password_hash FROM users WHERE username = :u"), {"u": username}
        ).fetchone()
    if row is None:
        return False
    return bcrypt.checkpw(password.encode(), row[0].encode())


def _register_user(connection, username: str, password: str) -> tuple[bool, str]:
    """Returns (success, error_message)."""
    import bcrypt
    if len(username.strip()) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    with connection.session as s:
        existing = s.execute(
            text("SELECT 1 FROM users WHERE username = :u"), {"u": username}
        ).fetchone()
        if existing:
            return False, f"Username '{username}' is already taken."
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        s.execute(
            text("INSERT INTO users (username, password_hash) VALUES (:u, :h)"),
            {"u": username, "h": hashed}
        )
        s.commit()
    return True, ""


# ==========================================
# "REMEMBER ME" TOKEN (survives tab reloads)
# ==========================================
# After login a signed token "<username>.<expiry>.<signature>" is stored in a browser cookie.
# On reload it is read via st.context.cookies and verified with HMAC, so users don't have to
# log in again until it expires. Streamlit can't set cookies itself, so a tiny JS snippet does it.

SESSION_TOKEN_DAYS = 14
SESSION_COOKIE     = "label_survey_session"


def _token_secret() -> bytes:
    """Signing key: `auth_token_secret` from secrets.toml if set, otherwise derived from the DB URL (also secret)."""
    secret = st.secrets.get("auth_token_secret") or st.secrets["connections"]["results_db"]["url"]
    return hashlib.sha256(f"label-survey-session:{secret}".encode()).digest()


def _make_session_token(username: str) -> str:
    user_b64 = base64.urlsafe_b64encode(username.encode()).decode().rstrip("=")
    expiry   = str(int(time.time()) + SESSION_TOKEN_DAYS * 86400)
    payload  = f"{user_b64}.{expiry}"
    sig      = hmac.new(_token_secret(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def _read_session_token(token: str) -> str | None:
    """Returns the username if the token is authentic and not expired, else None."""
    try:
        user_b64, expiry, sig = token.split(".")
        expected = hmac.new(_token_secret(), f"{user_b64}.{expiry}".encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected) or int(expiry) < time.time():
            return None
        return base64.urlsafe_b64decode(user_b64 + "=" * (-len(user_b64) % 4)).decode()
    except Exception:
        return None


def _write_session_cookie(token: str | None):
    """Sets (token) or deletes (None) the session cookie on the app's page via an invisible component."""
    max_age = SESSION_TOKEN_DAYS * 86400 if token else 0
    value   = token or ""
    components.html(f"""
        <script>
        const secure = window.parent.location.protocol === "https:" ? "; Secure" : "";
        window.parent.document.cookie =
            "{SESSION_COOKIE}={value}; path=/; max-age={max_age}; SameSite=Lax" + secure;
        </script>
    """, height=0)


def _user_exists(connection, username: str) -> bool:
    with connection.session as s:
        return s.execute(text("SELECT 1 FROM users WHERE username = :u"), {"u": username}).fetchone() is not None


# ==========================================
# AUTHENTICATION GATE
# ==========================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
if "show_register" not in st.session_state:
    st.session_state.show_register = False

# Cookie writes are queued by login/logout (which rerun immediately) and flushed here on the next run
if "cookie_action" in st.session_state:
    _write_session_cookie(st.session_state.pop("cookie_action"))

# Restore the login from the cookie after a tab reload. st.context.cookies is a snapshot taken when
# the tab connected, so after logging out we must ignore it for the rest of this session.
_cookie_token = st.context.cookies.get(SESSION_COOKIE)
if not st.session_state.authenticated and _cookie_token and not st.session_state.get("logged_out"):
    _token_user = _read_session_token(_cookie_token)
    if _token_user and _user_exists(get_db_connection(), _token_user):
        st.session_state.authenticated = True
        st.session_state.username = _token_user
    else:
        _write_session_cookie(None)   # expired or tampered: delete it and show the login form

if not st.session_state.authenticated:
    _conn_early = get_db_connection()
    _ensure_users_table(_conn_early)

    st.title("🔒 Login Required")
    
    with st.expander("ℹ️ About This Project & Why It Matters", expanded=True):
        st.markdown(
            """
            ### 🙏 Thank you for your participation
            #### 🌍 The Big Picture
            Our automated pipeline processes thousands of biomedical samples daily, attempting to extract 
            key categorical metadata (like tissue type, treatment, or developmental stage). 
            
            #### 🌱 How does this help CropXR?
            By evaluating these labels you help provide information on the accuracy of the labels in a  large Arabidopsis thaliana compendium. This compendium, together with the labels will allow researchers from our group \
            to have a large set of gene expression samples that they can easily filter and group by biological characteristics. We expect this to provide a knowledge base that can give our experimental results a broader biological context.

            #### 🎯 Your Role
            - **Quality Control:** Your manual verification provides the ground-truth dataset we need to catch subtle algorithmic blind spots.
            - **Downstream results validation:** Your input gives a measure of the accuracy of our labeling system. These scores are used later to estimate the accuracy of the labels and help interpret results from tasks such as GSEA, Differential expression, or cluster analisys.            
            *If you spot consistent systematic errors across multiple samples, please drop a note in the comments field.*
            
            #### 📝 What to do?
            - If this is your first time here create an account with a user name as password.
            - After that or if you have an account; Login and evaluate labels for samples.

            ### 🏆 Keep an eye on the leaderboard
            The top 3 participants at the end of the survey each get a €25 gift card! 🎁
            """
        )
    col_toggle1, col_toggle2 = st.columns([1, 1])
    with col_toggle1:
        if st.button("🔑 Login", use_container_width=True,
                     type="primary" if not st.session_state.show_register else "secondary"):
            st.session_state.show_register = False
            st.rerun()
    with col_toggle2:
        if st.button("📝 Create Account", use_container_width=True,
                     type="primary" if st.session_state.show_register else "secondary"):
            st.session_state.show_register = True
            st.rerun()

    st.divider()

    if not st.session_state.show_register:
        with st.form("login_form"):
            user_input = st.text_input("Username")
            pwd_input  = st.text_input("Password", type="password")
            submitted  = st.form_submit_button("Login", use_container_width=True)
            if submitted:
                if _verify_password(_conn_early, user_input.strip(), pwd_input):
                    st.session_state.authenticated = True
                    st.session_state.username = user_input.strip()
                    st.session_state.cookie_action = _make_session_token(user_input.strip())
                    st.session_state.logged_out = False
                    st.rerun()
                else:
                    st.error("Invalid username or password. Please try again.")
    else:
        st.write("Choose a username and password to get started.")
        with st.form("register_form"):
            new_user  = st.text_input("Choose a username")
            new_pwd   = st.text_input("Choose a password", type="password")
            new_pwd2  = st.text_input("Confirm password",  type="password")
            submitted = st.form_submit_button("Create Account", use_container_width=True)
            if submitted:
                if new_pwd != new_pwd2:
                    st.error("Passwords do not match.")
                else:
                    ok, err = _register_user(_conn_early, new_user.strip(), new_pwd)
                    if ok:
                        st.success(f"Account created! You can now log in as **{new_user.strip()}**.")
                        st.session_state.show_register = False
                        st.rerun()
                    else:
                        st.error(err)

    st.stop()

# ==========================================
# PERSISTENT SAVING (SQL)
# ==========================================

conn = get_db_connection()


def load_results():
    """Loads results from the SQL database (the only source of truth; the CSV is a write-only backup)."""
    try:
        with conn.session as s:
            s.execute(text("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id SERIAL PRIMARY KEY,
                    username TEXT, study_id TEXT, sample_id TEXT,
                    label_scores TEXT, comments TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            s.commit()

        df = conn.query("SELECT * FROM evaluations", ttl=0)

        return df if df is not None else pd.DataFrame(
            columns=["username", "study_id", "sample_id", "label_scores", "comments"]
        )
    except Exception as e:
        st.error(f"Database Error: {e}")
        return pd.DataFrame(columns=["username", "study_id", "sample_id", "label_scores", "comments"])


def save_evaluation(username, study_id, sample_id, label_scores, comments):
    """Saves to SQL and mirrors to local CSV."""
    with conn.session as s:
        s.execute(
            text("DELETE FROM evaluations WHERE username = :u AND study_id = :st AND sample_id = :sa"),
            params={"u": username, "st": study_id, "sa": sample_id}
        )
        s.execute(
            text("INSERT INTO evaluations (username, study_id, sample_id, label_scores, comments) "
                 "VALUES (:u, :st, :sa, :ls, :c)"),
            params={
                "u": username, "st": study_id, "sa": sample_id,
                "ls": json.dumps(label_scores), "c": comments.strip() if comments else ""
            }
        )
        s.commit()

    df = load_results()
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    df.to_csv(RESULTS_FILE, index=False)
    st.toast("Evaluation Saved", icon="💾")


# ==========================================
# SMART SAMPLE ASSIGNMENT
# ==========================================

def get_next_sample(all_sample_ids: list[str], username: str, results_df: pd.DataFrame, skip_ids: set[str] = None) -> str | None:
    if skip_ids is None:
        skip_ids = set()

    if not results_df.empty:
        user_done = set(results_df[results_df["username"] == username]["sample_id"].tolist())
    else:
        user_done = set()

    if not results_df.empty:
        other_evals = results_df[results_df["username"] != username]
        eval_counts = other_evals.groupby("sample_id").size().to_dict()
    else:
        eval_counts = {}

    def pick_from(candidates):
        non_skipped = [s for s in candidates if s not in skip_ids]
        skipped     = [s for s in candidates if s in skip_ids]
        pool = non_skipped if non_skipped else skipped
        return random.choice(pool) if pool else None

    eligible   = [s for s in all_sample_ids if s not in user_done]
    priority_2 = [s for s in eligible if eval_counts.get(s, 0) == 2]
    priority_1 = [s for s in eligible if eval_counts.get(s, 0) == 1]
    priority_0 = [s for s in eligible if eval_counts.get(s, 0) == 0]

    return (
        pick_from(priority_2)
        or pick_from(priority_1)
        or pick_from(priority_0)
    )


# ==========================================
# MAIN APPLICATION
# ==========================================

@st.cache_data
def load_survey_data():
    """Loads and combines RNA and MA sample sets, tagging their source."""
    combined_data = []
    
    # Load RNA samples
    if os.path.exists(PAYLOAD_FILE_RNA):
        with open(PAYLOAD_FILE_RNA, encoding="utf-8") as f:
            rna_data = json.load(f)
            for item in rna_data:
                item["source_type"] = "RNA"  # Tag origin
            combined_data.extend(rna_data)
            
    # Load Microarray samples
    if os.path.exists(PAYLOAD_FILE_MA):
        with open(PAYLOAD_FILE_MA, encoding="utf-8") as f:
            ma_data = json.load(f)
            for item in ma_data:
                item["source_type"] = "MA"  # Tag origin
            combined_data.extend(ma_data)
            
    return combined_data

data = load_survey_data()
if not data:
    st.error(f"Could not find any survey data files in './label_evaluation/data/'. Please ensure files exist.")
    st.stop()

all_sample_ids = [item["sample_id"] for item in data]


def _render_version_footer():
    """Tiny footer listing the label versions actually present in the loaded survey data."""
    versions = sorted({item.get("label_version", "unknown") for item in data})
    st.markdown(
        f"<div style='text-align:right; font-size:10px; opacity:0.4; margin-top:2em;'>"
        f"labels: {', '.join(versions)}</div>",
        unsafe_allow_html=True,
    )

sample_lookup  = {item["sample_id"]: item for item in data}

results_df        = load_results()
username          = st.session_state.username
user_results      = results_df[results_df["username"] == username] if not results_df.empty else pd.DataFrame()
evaluated_samples = user_results["sample_id"].unique().tolist() if not user_results.empty else []

# Session state for assignment and skips
if "assigned_sample_id" not in st.session_state:
    st.session_state.assigned_sample_id = None
if "skipped_ids" not in st.session_state:
    st.session_state.skipped_ids = set()
if "session_count" not in st.session_state:
    st.session_state.session_count = 0

if st.session_state.assigned_sample_id is None:
    st.session_state.assigned_sample_id = get_next_sample(
        all_sample_ids, username, results_df, st.session_state.skipped_ids
    )

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header(f"👤 User: {username}")
if st.sidebar.button("Logout"):
    st.session_state.authenticated       = False
    st.session_state.username            = ""
    st.session_state.assigned_sample_id  = None
    st.session_state.skipped_ids         = set()
    st.session_state.show_register       = False
    st.session_state.session_count       = 0
    st.session_state.cookie_action       = None   # delete the cookie on the next run
    st.session_state.logged_out          = True
    st.rerun()

st.sidebar.divider()

if not results_df.empty:
    other_eval_counts = results_df[results_df["username"] != username].groupby("sample_id").size().to_dict()
else:
    other_eval_counts = {}

remaining = sum(
    1 for s in all_sample_ids
    if s not in evaluated_samples and other_eval_counts.get(s, 0) < 3
)
st.sidebar.write(f"**Your Evaluations:** {len(evaluated_samples)}")
st.sidebar.write(f"**Samples Still Needing You:** {remaining}")
st.sidebar.write(f"**This Session:** {st.session_state.session_count}")

# Shared team goal: unique samples that have at least one evaluation from anyone
# The goal moves up to the next step once the current one is passed
team_done    = results_df["sample_id"].nunique() if not results_df.empty else 0
goals_passed = [g for g in TEAM_GOALS if team_done >= g]
team_goal    = next((g for g in TEAM_GOALS if team_done < g), TEAM_GOALS[-1])
st.sidebar.divider()
st.sidebar.subheader("🎯 Team Goal")
if goals_passed and team_done < TEAM_GOALS[-1]:
    st.sidebar.caption(f"✅ {goals_passed[-1]:,} reached — new target: {team_goal:,}!")
st.sidebar.progress(min(team_done / team_goal, 1.0))
if team_done >= TEAM_GOALS[-1]:
    st.sidebar.success(f"Final goal reached! {team_done:,} / {team_goal:,} samples reviewed 🎉")
else:
    st.sidebar.caption(f"{team_done:,} / {team_goal:,} samples reviewed — {team_goal - team_done:,} to go")

if not results_df.empty:
    st.sidebar.download_button(
        label="📥 Download Results (CSV Backup)",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="evaluation_results_combined_backup.csv",
        mime="text/csv",
        use_container_width=True,
    )
st.sidebar.divider()
st.sidebar.subheader("🏆 Leaderboard")

if not results_df.empty:
    # Count how many evaluations each user has done
    leaderboard_df = results_df["username"].value_counts().reset_index()
    leaderboard_df.columns = ["Researcher", "Evaluations"]
    
    # Optional: Highlight the current user with a medal or icon
    def format_researcher(row):
        name = row["Researcher"]
        if name == username:
            return f"{name} (You)"
        return name
        
    leaderboard_df["Researcher"] = leaderboard_df.apply(format_researcher, axis=1)
    
    # Display as a clean, index-less table in the sidebar
    st.sidebar.dataframe(
        leaderboard_df, 
        hide_index=True, 
        use_container_width=True
    )
else:
    st.sidebar.info("No evaluations yet. Be the first!")

# ── Milestone celebration (flagged by the submit handler before the rerun) ─────
milestone = st.session_state.pop("milestone_hit", None)
if milestone:
    st.balloons()
    st.toast(f"🎉 {milestone} evaluations — amazing work!", icon="🏅")

# ── Resolve current sample ─────────────────────────────────────────────────────
current_sample_id = st.session_state.assigned_sample_id

if current_sample_id is None:
    st.success("🎉 All done! Every sample has been evaluated enough times, or you've evaluated everything available.")
    _render_version_footer()
    st.stop()

current_sample = sample_lookup[current_sample_id]
study_id       = current_sample["study_id"]
sample_id      = current_sample["sample_id"]

# ── Header ─────────────────────────────────────────────────────────────────────

col1, col2 = st.columns([1, 2])

# ── LEFT COLUMN: Scoring form ──────────────────────────────────────────────────
with col1:
    st.subheader("📝 Algorithm Labels & Scoring")
    st.info(f"🧬 **Study ID:** `{study_id}` &nbsp;&nbsp;|&nbsp;&nbsp; 🆔 **Sample ID:** `{sample_id}`")
    # TULIP's "unspecified_notes" are model notes, not labels: hide them from reviewers (kept in the data)
    label_entries = [
        e for e in current_sample.get("label_entries", [])
        if not str(e.get("label_category", "")).startswith("unspecified_notes")
    ]

    prev_eval    = user_results[user_results["sample_id"] == sample_id] if not user_results.empty else pd.DataFrame()
    prev_scores  = {}
    default_comm = ""

    if not prev_eval.empty:
        with contextlib.suppress(Exception):
            prev_scores = json.loads(prev_eval.iloc[0]["label_scores"])
        default_comm = str(prev_eval.iloc[0]["comments"])
        if default_comm == "nan":
            default_comm = ""

    # Inject a clean snippet of CSS to explicitly paint the active button green
    st.markdown("""
        <style>
        /* Automatically styles the "Everything is Correct" button green when active */
        div[data-testid="stButton"] button:has(div p:contains("Everything is Correct")):not(:disabled) {
            background-color: #2e7d32 !important;
            color: white !important;
            border-color: #1b5e20 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    current_scores = {}

    if label_entries:
        st.markdown("**Evaluate Individual Categories:**")
        core_options = ["Correct", "Mostly Correct", "Incorrect"]
        
        for entry in label_entries:
            cat = entry.get("label_category", "unknown")
            val = entry.get("display_value", "unknown")
            # Show reviewers "no stress applied" instead of the raw "Control" treatment value
            if cat.startswith("treatment") and str(val).strip().lower() == "control":
                val = "no stress applied"

            # Smart default initialization
            default_val = prev_scores.get(cat, "Correct")
            try:
                idx = core_options.index(default_val)
            except ValueError:
                idx = 0
                    
            col_text, col_boxes = st.columns([1, 2])
            
            with col_text:
                st.markdown(f"<div style='padding-top: 10px;'>🔹 <b>{cat.capitalize()}</b>:<br><code>{val}</code></div>", unsafe_allow_html=True)
                
            with col_boxes:
                # Placed outside a form so changes register instantaneously
                current_scores[cat] = st.radio(
                    f"Accuracy for {cat}", 
                    options=core_options, 
                    index=idx,
                    horizontal=True,
                    key=f"rb_{sample_id}_{cat}",
                    label_visibility="collapsed"
                )
            
            st.markdown("<hr style='margin: 5px 0 15px 0; border: 0.5px solid #e6e6e6;'>", unsafe_allow_html=True)

        with st.expander("View Raw JSON Array (including intensities)"):
            st.json(label_entries)
    else:
        st.info("No labels found for this sample.")
        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

    # Added a unique template key so typed comments persist safely across layout refreshes
    comments = st.text_area("General Corrections / Comments:", value=default_comm, height=100, key=f"comm_{sample_id}")
    
    # Calculate if every row is clean
    all_correct = all(score == "Correct" for score in current_scores.values()) if current_scores else False

    st.markdown("<br>", unsafe_allow_html=True)
    btn_col1, btn_col2 = st.columns(2)
    
    with btn_col1:
        # Green button when active, automatically greys out when any selection changes
        all_correct_button = st.button(
            "✅ Everything is Correct",
            key=f"btn_correct_{sample_id}",
            disabled=not all_correct,
            use_container_width=True,
            type="primary" if all_correct else "secondary"
        )
        
    with btn_col2:
        # Standard highlighted submit button when changes or adjustments are active
        submit_button = st.button(
            "💾 Submit Sample",
            key=f"btn_submit_{sample_id}",
            disabled=all_correct,
            use_container_width=True,
            type="primary" if not all_correct else "secondary"
        )

    # Submission routing engine
    if all_correct_button or submit_button:
        if all_correct_button:
            for cat in current_scores:
                current_scores[cat] = "Correct"

        st.session_state.session_count += 1
        is_new_eval = sample_id not in evaluated_samples   # re-submitting an old sample doesn't raise the total
        new_total   = len(evaluated_samples) + int(is_new_eval)
        if is_new_eval and (new_total in MILESTONES or (new_total > 1000 and new_total % 500 == 0)):
            st.session_state.milestone_hit = new_total

        # Full-screen loading overlay: confirms the click and blocks double-submits. It stays
        # visible through the rerun until Streamlit redraws this slot with the next sample.
        st.markdown("""
            <style>
            .next-sample-overlay {
                position: fixed; inset: 0; z-index: 999999;
                display: flex; flex-direction: column; align-items: center; justify-content: center;
                background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(2px);
                font-size: 1.2rem; font-weight: 600; color: #1b5e20;
            }
            @media (prefers-color-scheme: dark) {
                .next-sample-overlay { background: rgba(14, 17, 23, 0.85); color: #81c784; }
            }
            .next-sample-spinner {
                width: 56px; height: 56px; margin-bottom: 18px; border-radius: 50%;
                border: 6px solid rgba(46, 125, 50, 0.2); border-top-color: #2e7d32;
                animation: next-sample-spin 0.8s linear infinite;
            }
            @keyframes next-sample-spin { to { transform: rotate(360deg); } }
            </style>
            <div class="next-sample-overlay">
                <div class="next-sample-spinner"></div>
                <div>💾 Sample #{session_count} this session — fetching the next one…</div>
            </div>
        """.replace("{session_count}", str(st.session_state.session_count)), unsafe_allow_html=True)

        _t0 = time.monotonic()
        save_evaluation(username, study_id, sample_id, current_scores, comments)
        # Keep the overlay up briefly even when saving is instant, so users notice it
        time.sleep(max(0.0, 0.8 - (time.monotonic() - _t0)))
        st.session_state.assigned_sample_id = None
        st.session_state.skipped_ids.discard(sample_id)
        st.rerun()

# ── RIGHT COLUMN: Metadata ─────────────────────────────────────────────────────
with col2:
    st.subheader("📄 Reference Metadata")


    # Dynamic Header based on Source
    source = current_sample.get("source_type", "Unknown")
    if source == "RNA":
        st.markdown("### 🔬 RNA-seq Sample Characteristics")
    elif source == "MA":
        st.markdown("### 🔬 Microarray Sample Characteristics")
    else:
        st.markdown("### 🔬 Sample Characteristics")
        
    chars = current_sample.get("characteristics", "No characteristics provided.")
    
    # Helper to parse and highlight sub-keys (e.g., "tissue: leaf" -> "**tissue**: leaf")
    def format_char_line(line: str) -> str:
        if ":" in line:
            key_part, val_part = line.split(":", 1)
            # Highlights the sub-key in green and bolds it
            return f"- :green[**{key_part.strip()}**]: {val_part.strip()}"
        return f"- {line.strip()}"

    # Generate processed lines
    processed_chars = []
    if isinstance(chars, str):
        processed_chars = [format_char_line(c) for c in chars.split("\n") if c.strip()]
    elif isinstance(chars, list):
        processed_chars = [format_char_line(c) for c in chars if str(c).strip()]
    
    if processed_chars:
        st.success("\n".join(processed_chars))
    else:
        st.success(str(chars))

    # Full Sample Metadata Section
    full_sample_meta = current_sample.get("full_sample_metadata", {})
    if full_sample_meta:
        with st.expander("🔍 View Full Sample Metadata", expanded=True):
            for key, val in full_sample_meta.items():
                if "characteristics" in key:
                    continue
                
                # Clean up array values safely
                if isinstance(val, list):
                    clean_val = ", ".join([str(v) for v in val]) if len(val) > 1 else str(val[0]) if val else ""
                else:
                    clean_val = str(val)
                
                # Make raw keys readable (e.g., "geo_accession" -> "Geo Accession")
                display_key = key.replace("_", " ").title()
                
                # Style the key using a distinct blue color block prefix
                st.markdown(f"🏷️ :blue[**{display_key}**]  \n&nbsp;&nbsp;&nbsp;&nbsp;{clean_val}")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📚 Study Context (Click to hide full study context)", expanded=True):
        # Using markdown block instead of raw text so it wraps beautifully and looks less harsh
        st.markdown(
            f"<div style='background-color: rgba(0,0,0,0.03); padding: 10px; border-radius: 5px; font-size: 14px;'>"
            f"{current_sample.get('study_context', 'No study context provided.')}"
            f"</div>", 
            unsafe_allow_html=True
        )

# ── Global results view ────────────────────────────────────────────────────────
with st.expander("View Global Saved Evaluations (Results CSV)"):
    st.dataframe(load_results())

_render_version_footer()
