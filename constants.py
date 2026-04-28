import os
import argparse as _argparse
import pandas as pd

def get_rna_used() -> bool:
    """Returns True if --rna was passed on the command line."""
    parser = _argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rna", action="store_true", default=False)
    args, _ = parser.parse_known_args()
    return args.rna
def get_matrix_used_for_training_bulk() -> str:
    parser = _argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train_matrix", default="rankin")

    args, _ = parser.parse_known_args()
    return args.train_matrix

GLOBAL_DIR_PATH = f"{os.getcwd()}/"
CLUSTER_RUN = GLOBAL_DIR_PATH != "/home/alex/Documents/GitHub/label_survey_website/"
EXPERIMENT_NAME = "5.0"

STORAGE_DIR = "/tudelft.net/staff-umbrella/GeneExpressionStorage/" if CLUSTER_RUN else "./new_storage/"

RNA_USED = get_rna_used()

DATA_IMPORT_DIR = "./data/downloads_new/"
GEO_DOWNLOAD_DIR = f"{DATA_IMPORT_DIR}geo_downloads_new/"
METADATA_OUTPUT_DIR = f"{DATA_IMPORT_DIR}metadata_new/"
PROCESSED_DATA_FOLDER = "new_storage/final_data/"

if CLUSTER_RUN:
    PROCESSED_DATA_FOLDER = "/tudelft.net/staff-umbrella/GeneExpressionStorage/final_data/"

COMBINED_DATA_OUTPUT_FILE = f"{PROCESSED_DATA_FOLDER}RMA_Microarray_Combined.csv"
CORE_DATA_DIR = "./data/core_data/"
SOFT_PATH = f"{CORE_DATA_DIR}old_geo_downloads/"
FIGURES_DIR = f"./outputs/{EXPERIMENT_NAME}{'_RNA' if RNA_USED else ''}/"

FILTERING_FIGURES = f"{FIGURES_DIR}filter_figures/{EXPERIMENT_NAME}/"
CLUSTER_EXPLORATION_FIGURES_DIR = f"{FIGURES_DIR}exploration_figures/"
MODEL = "TULIP_1.2"
if RNA_USED:
    MODEL = MODEL + "_RNA"
LABELS_PATH = f"{STORAGE_DIR}labels/{MODEL}/{EXPERIMENT_NAME}"
LABELS_PATH_RNA = f"{STORAGE_DIR}labels/{MODEL}_RNA/{EXPERIMENT_NAME}"
# Studies = ['GSE44053', 'GSE77815', 'GSE16474', 'GSE4062', 'GSE9415', 'GSE18624', 'GSE40061', 'GSE112161', 'GSE20494', 'GSE79997', 'GSE27552', 'GSE16222', 'GSE110857', 'GSE4760', 'GSE46205', 'GSE71001', 'GSE58616', 'GSE162310', 'GSE22107', 'GSE62163', 'GSE51897', 'GSE72949', 'GSE90562', 'GSE5628', 'GSE26266', 'GSE34188', 'GSE34595', 'GSE76827', 'GSE119383', 'GSE65046', 'GSE11758', 'GSE65414', 'GSE37408', 'GSE5624', 'GSE10643', 'GSE15577', 'GSE11538', 'GSE70861', 'GSE6583', 'GSE27551', 'GSE12619', 'GSE121225', 'GSE108070', 'GSE78713', 'GSE110079', 'GSE63128', 'GSE60960', 'GSE37118', 'GSE79681', 'GSE63372', 'GSE5622', 'GSE26983', 'GSE27550', 'GSE19603', 'GSE95202', 'GSE53308', 'GSE16765', 'GSE71855', 'GSE58620', 'GSE24177', 'GSE35258', 'GSE10670', 'GSE49418', 'GSE18666', 'GSE83136', 'GSE44655', 'GSE27549', 'GSE19700', 'GSE103398', 'GSE63522', 'GSE201609', 'GSE5620', 'GSE66369', 'GSE2268', 'GSE71237', 'GSE48474', 'GSE41935', 'GSE27548', 'GSE5623', 'GSE72050', 'GSE126373']  # noqa: E501
