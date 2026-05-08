"""
Data Ingestion Module — loads and validates the raw water potability dataset.

Each feature's WHO/EPA significance:
- ph: Acidity/alkalinity. WHO safe range 6.5–8.5.
- Hardness: Calcium carbonate (mg/L). High >300 linked to scale buildup.
- Solids: Total dissolved solids (ppm). >1000 ppm unpalatable per WHO.
- Chloramines: Disinfectant level (ppm). EPA MRDL 4 mg/L.
- Sulfate: Sulfate ion (mg/L). Laxative effect >500 mg/L.
- Conductivity: Electrical conductivity (μS/cm). Proxy for ionic content.
- Organic_carbon: Organic carbon (ppm). Indicates biological contamination.
- Trihalomethanes: DBP byproducts (μg/L). Carcinogenic; EPA MCL 80 μg/L.
- Turbidity: Cloudiness (NTU). WHO <4 NTU; harbours pathogens.
"""
from pathlib import Path
from typing import Optional
import mlflow
import pandas as pd
from src.config import FEATURE_COLUMNS, TARGET_COLUMN, RAW_CSV_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw water potability CSV and perform schema validation.

    Args:
        filepath: Path to the CSV file. Defaults to config.RAW_CSV_PATH.

    Returns:
        A pd.DataFrame with validated columns.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If required columns missing or Potability not binary.
    """
    filepath = filepath or RAW_CSV_PATH
    if not filepath.exists():
        raise FileNotFoundError(
            f"Raw data not found at {filepath}. "
            "Place water_potability.csv in data/raw/"
        )
    logger.info(f"Loading raw dataset from {filepath}")
    df = pd.read_csv(filepath)

    # Column validation
    expected = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {sorted(df.columns)}")

    # Target validation — must be binary {0, 1}
    unique_targets = set(df[TARGET_COLUMN].unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"Target must be binary {{0,1}}. Found: {unique_targets}")

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Null counts:\n{df.isnull().sum()}")
    logger.info(f"Class balance:\n{df[TARGET_COLUMN].value_counts()}")
    return df


def log_dataset_metadata(df: pd.DataFrame) -> None:
    """Log dataset metadata to MLflow as parameters.

    Args:
        df: The loaded DataFrame to describe.
    """
    nulls = df.isnull().sum()
    counts = df[TARGET_COLUMN].value_counts()
    mlflow.log_params({
        "dataset_rows": df.shape[0],
        "dataset_cols": df.shape[1],
        "null_ph": int(nulls.get("ph", 0)),
        "null_sulfate": int(nulls.get("Sulfate", 0)),
        "null_trihalomethanes": int(nulls.get("Trihalomethanes", 0)),
        "class_0_count": int(counts.get(0, 0)),
        "class_1_count": int(counts.get(1, 0)),
    })
    logger.info("Dataset metadata logged to MLflow")


def download_dataset() -> Path:
    """Download the water potability dataset from Kaggle if not present.

    Returns:
        Path to the downloaded CSV file.

    Raises:
        RuntimeError: If download fails.
    """
    if RAW_CSV_PATH.exists():
        logger.info("Dataset already exists, skipping download")
        return RAW_CSV_PATH

    logger.info("Downloading water_potability.csv from Kaggle...")
    try:
        import kagglehub, shutil
        path = kagglehub.dataset_download("adityakadiwal/water-potability")
        for f in Path(path).rglob("water_potability.csv"):
            shutil.copy2(f, RAW_CSV_PATH)
            logger.info(f"Dataset saved to {RAW_CSV_PATH}")
            return RAW_CSV_PATH
        raise FileNotFoundError(f"CSV not found in: {path}")
    except Exception as e:
        logger.warning(f"Download failed: {e}")
        raise RuntimeError(
            "Could not download dataset. Please manually download "
            "'water_potability.csv' from Kaggle and place it in data/raw/"
        )
