"""Central configuration for the Shill Bidding Detection project.

All paths are resolved relative to the project root so the project can be
moved or cloned anywhere without editing code.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
BACKLOG_DIR = DATA_DIR / "backlog"

MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
EVALUATION_DIR = MODELS_DIR / "evaluation"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"

RAW_DATA_FILE = RAW_DATA_DIR / "shill_bidding_raw.csv"
RAW_METADATA_FILE = RAW_DATA_DIR / "uci_metadata.json"
CLEAN_DATA_FILE = PROCESSED_DATA_DIR / "shill_bidding_clean.csv"
SPLIT_FILE = PROCESSED_DATA_DIR / "train_test_split.csv"

MODEL_BUNDLE_FILE = TRAINED_MODELS_DIR / "best_model.joblib"
ALL_MODELS_FILE = TRAINED_MODELS_DIR / "all_candidate_models.joblib"
EVALUATION_RESULTS_FILE = EVALUATION_DIR / "evaluation_results.json"
CURVES_FILE = EVALUATION_DIR / "curves.joblib"
EDA_SUMMARY_FILE = EVALUATION_DIR / "eda_summary.json"

MODEL_COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"
VALIDATION_STUDY_CSV = REPORTS_DIR / "validation_strategy_study.csv"
MODEL_REPORT_MD = REPORTS_DIR / "model_comparison_report.md"
DATASET_REPORT_MD = REPORTS_DIR / "dataset_analysis_report.md"

BACKLOG_DB_FILE = BACKLOG_DIR / "fraud_backlog.db"

# --------------------------------------------------------------------------- #
# Dataset (UCI Machine Learning Repository, ID 562)
# --------------------------------------------------------------------------- #
UCI_DATASET_ID = 562
UCI_DATASET_URL = "https://archive.ics.uci.edu/dataset/562/shill+bidding+dataset"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/562/shill+bidding+dataset.zip"
UCI_METADATA_API = "https://archive.ics.uci.edu/api/dataset?id=562"
UCI_DOI = "https://doi.org/10.24432/C5Z611"
DATASET_LICENSE = "CC BY 4.0"
# Citation exactly as registered with DataCite for the DOI (author recorded as "Unknown").
DATASET_CITATION = (
    "Shill Bidding Dataset (2020). UCI Machine Learning Repository. "
    "https://doi.org/10.24432/C5Z611"
)

# Column names exactly as published in the official UCI CSV.
RECORD_ID = "Record_ID"
AUCTION_ID = "Auction_ID"
BIDDER_ID = "Bidder_ID"
TARGET = "Class"

IDENTIFIER_COLUMNS = [RECORD_ID, AUCTION_ID, BIDDER_ID]

BEHAVIOURAL_FEATURES = [
    "Bidder_Tendency",
    "Bidding_Ratio",
    "Successive_Outbidding",
    "Last_Bidding",
    "Auction_Bids",
    "Starting_Price_Average",
    "Early_Bidding",
    "Winning_Ratio",
    "Auction_Duration",
]

EXPECTED_COLUMNS = IDENTIFIER_COLUMNS + BEHAVIOURAL_FEATURES + [TARGET]

# Level at which each behavioural feature is measured. Bidder-level features
# are aggregated over all auctions of a bidder; auction-level features are
# shared by every bidder of the same auction.
FEATURE_LEVEL = {
    "Bidder_Tendency": "bidder (across auctions)",
    "Bidding_Ratio": "bidder within auction",
    "Successive_Outbidding": "bidder within auction",
    "Last_Bidding": "bidder within auction",
    "Auction_Bids": "auction",
    "Starting_Price_Average": "auction",
    "Early_Bidding": "bidder within auction",
    "Winning_Ratio": "bidder (across auctions)",
    "Auction_Duration": "auction",
}

# Descriptions follow the official UCI variable information.
FEATURE_DESCRIPTIONS = {
    "Record_ID": "Unique identifier of a record (identifier, not a predictor).",
    "Auction_ID": "Unique identifier of an auction (identifier, used for grouping).",
    "Bidder_ID": "Anonymised bidder identifier (identifier, used for grouping).",
    "Bidder_Tendency": (
        "A shill bidder participates exclusively in auctions of few sellers rather "
        "than a diversified lot - a collusive act between seller and accomplice."
    ),
    "Bidding_Ratio": (
        "A shill bidder participates more frequently to raise the auction price "
        "and attract higher bids from legitimate participants."
    ),
    "Successive_Outbidding": (
        "A shill bidder successively outbids himself even though he is the current "
        "winner, increasing the price gradually with small consecutive increments."
    ),
    "Last_Bidding": (
        "A shill bidder becomes inactive at the last stage of the auction "
        "(more than 90% of the auction duration) to avoid winning the auction."
    ),
    "Auction_Bids": (
        "Auctions with shill-bidding activity tend to have a much higher number of "
        "bids than the average of bids in concurrent auctions."
    ),
    "Starting_Price_Average": (
        "Auction starting price: a shill bidder's seller usually offers a small "
        "starting price to attract legitimate bidders into the auction."
    ),
    "Early_Bidding": (
        "A shill bidder tends to bid pretty early in the auction (less than 25% of "
        "the auction duration) to get the attention of auction users."
    ),
    "Winning_Ratio": (
        "A shill bidder competes in many auctions but hardly wins any auctions."
    ),
    "Auction_Duration": "How long the auction lasted (in days).",
    "Class": "0 = normal bidding behaviour; 1 = shill bidding behaviour.",
}

CLASS_NAMES = {0: "Normal", 1: "Potential shill bidding"}

# --------------------------------------------------------------------------- #
# Experiment settings
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42
TEST_SIZE_FOLDS = 5          # 1 of 5 grouped folds -> ~20 % held-out test set
CV_FOLDS = 5
# Primary grouping for splitting: unseen auctions. See README for rationale.
PRIMARY_GROUP = AUCTION_ID
# Correlation level above which two features are reported as redundant.
CORRELATION_THRESHOLD = 0.80
# Models within this CV PR-AUC distance of the best are considered tied; the
# simplest (most interpretable) tied model is then preferred.
SELECTION_TOLERANCE = 0.005
# Threshold optimisation: maximise F-beta on out-of-fold training predictions.
THRESHOLD_BETA = 2.0  # recall weighted twice as heavily as precision
MIN_PRECISION_FOR_THRESHOLD = 0.80

# --------------------------------------------------------------------------- #
# Backlog
# --------------------------------------------------------------------------- #
REVIEW_STATUSES = [
    "Pending Review",
    "Under Investigation",
    "Confirmed Suspicious",
    "False Positive",
    "Closed",
]
DEFAULT_REVIEW_STATUS = "Pending Review"


def ensure_directories() -> None:
    """Create every directory the project writes to (idempotent)."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        BACKLOG_DIR,
        TRAINED_MODELS_DIR,
        EVALUATION_DIR,
        FIGURES_DIR,
        LOGS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
