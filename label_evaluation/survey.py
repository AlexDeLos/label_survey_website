import contextlib
import json
import os
import random
import pandas as pd
import streamlit as st
import sys
from sqlalchemy import text

module_dir = "./"
sys.path.append(module_dir)

from constants import RNA_USED  # noqa: E402

# ==========================================
# CONFIGURATION & AUTHENTICATION
# ==========================================

PAYLOAD_FILE = f"./label_evaluation/data/survey_data_{'RNA' if RNA_USED else 'MA'}.json"
RESULTS_FILE = f"./label_evaluation/data/evaluation_results_{'RNA' if RNA_USED else 'MA'}.csv"

# Barebones Authentication Dictionary
VALID_USERS = {"researcher1": "pass123", "researcher2": "tulip2026", "alex": "admin"}

st.set_page_config(layout="wide", page_title="Label Evaluation Tool")

# --- Authentication Logic ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""

if not st.session_state.authenticated:
    st.title("🔒 Login Required")
    st.write("Please log in to access the Label Evaluation Tool.")

    with st.form("login_form"):
        user_input = st.text_input("Username")
        pwd_input = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if user_input in VALID_USERS and VALID_USERS[user_input] == pwd_input:
                st.session_state.authenticated = True
                st.session_state.username = user_input
                st.rerun()
            else:
                st.error("Invalid username or password. Please try again.")
    st.stop()

# ==========================================
# PERSISTENT SAVING SOLUTIONS (SQL)
# ==========================================

conn = st.connection("results_db", type="sql")

def load_results():
    """Loads results from SQL database, falling back to CSV if DB is empty."""
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

        if (df is None or df.empty) and os.path.exists(RESULTS_FILE):
            df = pd.read_csv(RESULTS_FILE)
            if "username" not in df.columns:
                df["username"] = "unknown_legacy_user"
            return df

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
    """
    Returns the next sample_id this user should evaluate, based on:
      - Priority 1: samples with exactly 2 evaluations by OTHER users
      - Priority 2: samples with exactly 1 evaluation by OTHER users
      - Priority 3: samples with 0 evaluations by anyone
    
    Hard constraints:
      - The current user must never see a sample they have already evaluated.
      - skip_ids: samples the user has explicitly skipped this session (excluded for now,
        but will be reconsidered once higher-priority work runs out).
    """
    if skip_ids is None:
        skip_ids = set()

    # Samples this user has already evaluated — permanently excluded
    if not results_df.empty:
        user_done = set(results_df[results_df["username"] == username]["sample_id"].tolist())
    else:
        user_done = set()

    # Count evaluations per sample by OTHER users only
    if not results_df.empty:
        other_evals = results_df[results_df["username"] != username]
        eval_counts = other_evals.groupby("sample_id").size().to_dict()
    else:
        eval_counts = {}

    def pick_from(candidates):
        """Randomly pick from candidates, preferring non-skipped. Falls back to skipped if nothing else."""
        non_skipped = [s for s in candidates if s not in skip_ids]
        skipped     = [s for s in candidates if s in skip_ids]
        pool = non_skipped if non_skipped else skipped
        return random.choice(pool) if pool else None

    # Build candidate pools (excluding samples user already evaluated)
    eligible = [s for s in all_sample_ids if s not in user_done]

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
    if not os.path.exists(PAYLOAD_FILE):
        return []
    with open(PAYLOAD_FILE, encoding="utf-8") as f:
        return json.load(f)


st.title("🧬 Sample Label Evaluator")

data = load_survey_data()
if not data:
    st.error(f"Could not find '{PAYLOAD_FILE}'. Please ensure the file exists.")
    st.stop()

all_sample_ids = [item["sample_id"] for item in data]
sample_lookup  = {item["sample_id"]: item for item in data}

results_df = load_results()
username   = st.session_state.username

user_results     = results_df[results_df["username"] == username] if not results_df.empty else pd.DataFrame()
evaluated_samples = user_results["sample_id"].unique().tolist() if not user_results.empty else []

# --- Session state: track current assignment and skipped samples ---
if "assigned_sample_id" not in st.session_state:
    st.session_state.assigned_sample_id = None
if "skipped_ids" not in st.session_state:
    st.session_state.skipped_ids = set()

# Assign a sample if we don't have one yet (or the previous one was just saved/skipped)
if st.session_state.assigned_sample_id is None:
    st.session_state.assigned_sample_id = get_next_sample(
        all_sample_ids, username, results_df, st.session_state.skipped_ids
    )

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header(f"👤 User: {username}")
if st.sidebar.button("Logout"):
    st.session_state.authenticated    = False
    st.session_state.username         = ""
    st.session_state.assigned_sample_id = None
    st.session_state.skipped_ids      = set()
    st.rerun()

st.sidebar.divider()

# Progress: count samples where this user has NOT yet evaluated AND there are
# still < 3 other-user evaluations (i.e. work remaining for this user)
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

if not results_df.empty:
    st.sidebar.download_button(
        label="📥 Download Results (CSV Backup)",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="evaluation_results_backup.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ── Resolve current sample ─────────────────────────────────────────────────────
current_sample_id = st.session_state.assigned_sample_id

if current_sample_id is None:
    st.success("🎉 All done! Every sample has been evaluated enough times, or you've evaluated everything available.")
    st.stop()

current_sample = sample_lookup[current_sample_id]
study_id       = current_sample["study_id"]
sample_id      = current_sample["sample_id"]

# ── Header ─────────────────────────────────────────────────────────────────────
st.header(f"Evaluating: {sample_id} ({study_id})")

# Show how many other evaluations this sample already has (transparency)
other_count = other_eval_counts.get(sample_id, 0)
st.caption(f"This sample has been evaluated by {other_count} other reviewer(s) so far.")

_, nav_col_skip = st.columns([3, 1])
with nav_col_skip:
    if st.button("Skip Sample ⏭️", use_container_width=True):
        st.session_state.skipped_ids.add(sample_id)
        st.session_state.assigned_sample_id = get_next_sample(
            all_sample_ids, username, results_df, st.session_state.skipped_ids
        )
        st.rerun()

st.divider()

col1, col2 = st.columns([1, 1.5])

# ── LEFT COLUMN: Scoring form ──────────────────────────────────────────────────
with col1:
    st.subheader("📝 Algorithm Labels & Scoring")
    label_entries = current_sample.get("label_entries", [])

    prev_eval   = user_results[user_results["sample_id"] == sample_id] if not user_results.empty else pd.DataFrame()
    prev_scores = {}
    default_comm = ""

    if not prev_eval.empty:
        with contextlib.suppress(Exception):
            prev_scores = json.loads(prev_eval.iloc[0]["label_scores"])
        default_comm = str(prev_eval.iloc[0]["comments"])
        if default_comm == "nan":
            default_comm = ""

    acc_options = ["Select...", "Correct", "Mostly Correct", "Incorrect"]

    with st.form(key=f"form_{sample_id}"):
        current_scores = {}

        if label_entries:
            st.markdown("**Evaluate Each Label Category:**")
            for entry in label_entries:
                cat = entry.get("label_category", "unknown")
                val = entry.get("display_value", "unknown")
                st.markdown(f"🔹 **{cat.capitalize()}**: `{val}`")
                default_val = prev_scores.get(cat, "Select...")
                try:
                    idx = acc_options.index(default_val)
                except ValueError:
                    idx = 0
                current_scores[cat] = st.selectbox(
                    f"Accuracy for {cat}", options=acc_options, index=idx,
                    key=f"sb_{sample_id}_{cat}", label_visibility="collapsed"
                )
                st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

            with st.expander("View Raw JSON Array (including intensities)"):
                st.json(label_entries)
        else:
            st.info("No labels found for this sample.")
            st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

        comments      = st.text_area("General Corrections / Comments:", value=default_comm, height=100)
        submit_button = st.form_submit_button(label="💾 Save & Next", use_container_width=True)

        if submit_button:
            unanswered = [cat for cat, score in current_scores.items() if score == "Select..."]
            if unanswered:
                st.error(f"Please select an accuracy rating for: **{', '.join(unanswered)}**")
            else:
                save_evaluation(username, study_id, sample_id, current_scores, comments)
                # Clear assignment so the next sample is freshly computed after save
                st.session_state.assigned_sample_id = None
                # Remove from skipped if it was there (it's now done)
                st.session_state.skipped_ids.discard(sample_id)
                st.rerun()

# ── RIGHT COLUMN: Metadata ─────────────────────────────────────────────────────
with col2:
    st.subheader("📄 Reference Metadata")
    st.info(f"**Study ID:** `{study_id}` &nbsp; | &nbsp; **Sample ID:** `{sample_id}`")

    st.markdown("### 🔬 Sample Characteristics")
    chars = current_sample.get("characteristics", "No characteristics provided.")
    if isinstance(chars, str):
        char_lines = chars.split("\n")
        st.success("\n".join([f"- {c}" for c in char_lines if c.strip()]))
    elif isinstance(chars, list):
        st.success("\n".join([f"- {c}" for c in chars]))
    else:
        st.success(chars)

    full_sample_meta = current_sample.get("full_sample_metadata", {})
    if full_sample_meta:
        with st.expander("🔍 View Full Sample Metadata"):
            for key, val in full_sample_meta.items():
                if "characteristics" in key:
                    continue
                if isinstance(val, list) and len(val) == 1:
                    clean_val = val[0]
                elif isinstance(val, list):
                    clean_val = ", ".join([str(v) for v in val])
                else:
                    clean_val = val
                st.markdown(f"**{key}**: {clean_val}")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📚 Study Context (Click to hide full study context)", expanded=True):
        st.text(current_sample.get("study_context", "No study context provided."))

# ── Global results view ────────────────────────────────────────────────────────
with st.expander("View Global Saved Evaluations (Results CSV)"):
    st.dataframe(load_results())