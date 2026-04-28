import glob
import json
import os

import pandas as pd
import streamlit as st

# Configuration
# Point these to where your actual label and metadata JSON files are stored
LABELS_DIR = "./new_storage/labels/TULIP_1.2_RNA/4.5"  # e.g., folder containing GSE40216.json
METADATA_DIR = "./new_storage/rnaseq_data/metadata/GSE40216/"  # e.g., folder containing GSE163009_GSM4970099.json
RESULTS_FILE = "./label_evaluation/evaluation_results.csv"

st.set_page_config(layout="wide", page_title="Label Evaluation Tool")


# --- Helper Functions ---
@st.cache_data
def load_all_label_files():
    """Loads all label JSON files into a flat list of samples to evaluate."""
    samples = []
    label_files = glob.glob(os.path.join(LABELS_DIR, "*.json"))

    for file_path in label_files:
        study_id = os.path.basename(file_path).split(".")[0]
        try:
            with open(file_path) as f:
                data = json.load(f)

                # Format 1: Standard Dictionary (e.g., GSE40216.json)
                # {"GSM123": {"tissue": ...}, "GSM124": {"tissue": ...}}
                if isinstance(data, dict):
                    for sample_id, labels in data.items():
                        samples.append({"study_id": study_id, "sample_id": sample_id, "labels": labels})

                # Format 2: List of objects (e.g., tulip_condensed_labels.json)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            # Sub-format 2A: The new condensed format
                            # [{"tissue": [...], "id": "GSM123"}]
                            if "id" in item:
                                sample_id = item["id"]
                                # Grab everything EXCEPT the 'id' key to use as labels
                                labels = {k: v for k, v in item.items() if k != "id"}
                                samples.append(
                                    {
                                        "study_id": study_id,  # This will be 'tulip_condensed_labels'
                                        "sample_id": sample_id,
                                        "labels": labels,
                                    }
                                )

                            # Sub-format 2B: Alternative list format
                            # [{"GSM123": {"tissue": ...}}]
                            else:
                                for sample_id, labels in item.items():
                                    samples.append({"study_id": study_id, "sample_id": sample_id, "labels": labels})
                else:
                    print(f"Warning: {file_path} contains unexpected data format: {type(data)}")

        except json.JSONDecodeError:
            print(f"Error reading {file_path}: File is not valid JSON.")

    return samples


def load_metadata(sample_id):
    """Finds and loads the metadata JSON for a specific sample."""
    # Assuming metadata files contain the sample_id in the name
    meta_files = glob.glob(os.path.join(METADATA_DIR, f"*{sample_id}*.json"))
    if not meta_files:
        return {"error": f"No metadata file found for {sample_id}"}

    with open(meta_files[0]) as f:
        return json.load(f)


def load_results():
    if os.path.exists(RESULTS_FILE):
        return pd.read_csv(RESULTS_FILE)
    return pd.DataFrame(columns=["study_id", "sample_id", "label_category", "predicted_value", "score"])


def save_evaluations(study_id, sample_id, evaluations):
    df = load_results()
    # Remove older evaluations for this sample if re-evaluating
    df = df[~((df["study_id"] == study_id) & (df["sample_id"] == sample_id))]
    # Append new evaluations
    new_rows = pd.DataFrame(evaluations)
    df = pd.concat([df, new_rows], ignore_index=True)
    df.to_csv(RESULTS_FILE, index=False)


# --- App Layout ---
st.title("🧬 Sample Label Evaluator")

# 1. Load Data
samples = load_all_label_files()
if not samples:
    st.warning(f"No label files found in '{LABELS_DIR}'. Please check the directory.")
    st.stop()

results_df = load_results()
evaluated_samples = results_df["sample_id"].unique().tolist()

# 2. Sidebar for Navigation
st.sidebar.header("Navigation")
total_samples = len(samples)
st.sidebar.write(f"**Progress:** {len(evaluated_samples)} / {total_samples} Evaluated")

# Let researcher select a sample (mark evaluated ones with a checkmark)
sample_options = [f"{s['study_id']} - {s['sample_id']} {'✅' if s['sample_id'] in evaluated_samples else ''}" for s in samples]
selected_option = st.sidebar.selectbox("Select Sample to Evaluate:", sample_options)

# Get current sample data
selected_index = sample_options.index(selected_option)
current_sample = samples[selected_index]
study_id = current_sample["study_id"]
sample_id = current_sample["sample_id"]
labels = current_sample["labels"]
metadata = load_metadata(sample_id)

st.header(f"Evaluating: {sample_id} ({study_id})")

# 3. Main Interface (Two Columns)
col1, col2 = st.columns([1, 1.5])  # Left column slightly narrower than right

with col1:
    st.subheader("📝 Algorithm Labels & Scoring")
    st.markdown("""
    **Scoring Key:**
    * **0**: Correct
    * **1**: Mostly Correct
    * **2**: Incorrect
    """)

    evaluations_to_save = []

    with st.form(key=f"form_{sample_id}"):
        # Dynamically create scoring buttons for every label present
        for label_category, predicted_value in labels.items():
            st.markdown(f"**{label_category.capitalize()}**")
            # Format the predicted value for display (handling dicts like treatment intensity)
            display_val = str(predicted_value)
            st.info(f"Predicted: `{display_val}`")

            # Default to None, or pre-load previous score if it exists
            score = st.radio(f"Score for {label_category}", options=[0, 1, 2], index=0, horizontal=True, key=f"radio_{sample_id}_{label_category}")

            evaluations_to_save.append({"study_id": study_id, "sample_id": sample_id, "label_category": label_category, "predicted_value": display_val, "score": score})
            st.divider()

        submit_button = st.form_submit_button(label="Save Evaluation")
        if submit_button:
            save_evaluations(study_id, sample_id, evaluations_to_save)
            st.success(f"Successfully saved evaluation for {sample_id}!")
            # st.rerun() # Uncomment to auto-refresh state after saving in newer Streamlit versions

with col2:
    st.subheader("📄 Reference Metadata")
    if "error" in metadata:
        st.error(metadata["error"])
        st.markdown("*Make sure the metadata JSON file exists in your metadata directory.*")
    else:
        # Helper function to clean up lists (e.g. ["text"] -> "text")
        def format_meta_value(val):
            if isinstance(val, list):
                if len(val) == 1:
                    return str(val[0])
                return "\n".join([f"- {v!s}" for v in val])
            return str(val)

        # 1. Top Level Info
        st.info(f"**Study ID:** `{metadata.get('study_id', 'N/A')}`   |   **Sample ID:** `{metadata.get('sample_id', 'N/A')}`   |   **Platform:** `{metadata.get('platform', 'N/A')}`")

        # 2. Sample Metadata (Most important for labeling)
        if "sample_metadata" in metadata:
            st.markdown("### 🔬 Sample Metadata")

            # Highlight 'characteristics_ch1' if it exists, as it usually contains the actual labels
            sample_meta = metadata["sample_metadata"]
            if "characteristics_ch1" in sample_meta:
                st.success("**Characteristics:**\n" + format_meta_value(sample_meta["characteristics_ch1"]))

            # Print the rest of the sample metadata
            for key, val in sample_meta.items():
                if key != "characteristics_ch1":  # skip since we already printed it
                    st.markdown(f"**{key}**: {format_meta_value(val)}")

        # 3. Study Metadata (Put in an expander to save space)
        if "study_metadata" in metadata:
            st.markdown("<br>", unsafe_allow_html=True)  # little spacing
            with st.expander("📚 Study Metadata (Click to view full study context)"):
                for key, val in metadata["study_metadata"].items():
                    st.markdown(f"**{key}**:\n {format_meta_value(val)}")
                    st.markdown("---")

# 4. View Results
with st.expander("View Saved Evaluations (Results CSV)"):
    st.dataframe(load_results())
