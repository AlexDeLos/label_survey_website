#!/bin/bash
# Builds survey_data_MA.json and survey_data_RNA.json from the TULIP_2.0.0 / 7.0 labels,
# inside an apptainer container. Run from anywhere:
#   ./label_evaluation/apptainer/prepare_survey_data.sh
# Rebuild the image after changing requirements.txt:
#   REBUILD=1 ./label_evaluation/apptainer/prepare_survey_data.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
STORAGE_DIR="${GENE_STORAGE_DIR:-/home/alex/Documents/GitHub/Dataset_fusion_Microarray/new_storage}"
SIF="$SCRIPT_DIR/label_survey.sif"

if [[ ! -f "$SIF" || -n "${REBUILD:-}" ]]; then
    echo "=== Building $SIF ==="
    (cd "$REPO_DIR" && apptainer build --force "$SIF" label_evaluation/apptainer/label_survey.def)
fi

run() {
    apptainer exec \
        --bind "$STORAGE_DIR":"$STORAGE_DIR" \
        --bind "$REPO_DIR":/repo \
        --pwd /repo \
        --env GENE_STORAGE_DIR="$STORAGE_DIR/" \
        "$SIF" \
        python -u label_evaluation/1_prepare_survey_data.py "$@"
}

echo "=== Microarray labels ==="
run
echo "=== RNA-seq labels ==="
run --rna
