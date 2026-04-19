"""LIAR dataset loading and normalization utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
from milestone2.logger import logger_rag
from milestone2.config import FACT_CHECK_DATA_PATH


class LIARDatasetLoader:
    DEFAULT_FILENAME = "liar_dataset/train.tsv"
    SUPPORTED_TEXT_COLUMNS = ["statement", "claim", "text", "description"]
    SUPPORTED_LABEL_COLUMNS = ["label", "truth", "veracity", "rating"]

    def __init__(self, dataset_path: Optional[Path] = None):
        self.dataset_path = Path(dataset_path) if dataset_path else FACT_CHECK_DATA_PATH / self.DEFAULT_FILENAME

    def download_dataset(self, url: str, destination: Optional[Path] = None) -> Path:
        """Download a dataset file from a URL to the fact-check data directory."""
        destination_path = Path(destination) if destination else self.dataset_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        logger_rag.info(f"Downloading LIAR dataset from: {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with open(destination_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)

        logger_rag.info(f"LIAR dataset downloaded to: {destination_path}")
        return destination_path

    def load_dataset(self, dataset_path: Optional[Path] = None) -> List[Dict[str, str]]:
        """Load the LIAR dataset from TSV and normalize records."""
        path = Path(dataset_path) if dataset_path else self.dataset_path
        logger_rag.info(f"Loading LIAR dataset from: {path}")

        if not path.exists():
            raise FileNotFoundError(f"LIAR dataset file not found at {path}")

        try:
            # LIAR dataset is TSV format with no headers
            df = pd.read_csv(path, sep='\t', header=None, dtype=str, keep_default_na=False)
            # Set column names based on LIAR dataset format
            df.columns = [
                'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
                'state', 'party', 'barely_true_count', 'false_count', 'half_true_count',
                'mostly_true_count', 'pants_fire_count', 'context'
            ]
        except Exception as exc:
            logger_rag.error(f"Failed to read LIAR dataset TSV: {exc}")
            raise

        records = self._normalize_dataframe(df)
        logger_rag.info(f"Loaded {len(records)} LIAR records")
        return records

    def _normalize_dataframe(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        columns = {c.lower(): c for c in df.columns}

        text_col = self._select_column(columns, self.SUPPORTED_TEXT_COLUMNS)
        label_col = self._select_column(columns, self.SUPPORTED_LABEL_COLUMNS)

        if not text_col:
            raise ValueError(
                "LIAR dataset must contain one of the following text columns: "
                f"{', '.join(self.SUPPORTED_TEXT_COLUMNS)}"
            )

        if not label_col:
            label_col = None
            logger_rag.warning("No label column found in LIAR dataset; labels will be empty")

        records: List[Dict[str, str]] = []
        for index, row in df.iterrows():
            claim_text = str(row[text_col]).strip()
            if not claim_text:
                continue

            label_text = str(row[label_col]).strip() if label_col else ""
            metadata = {
                "id": str(row.get("id", index)).strip() or str(index),
                "source": str(row.get("source", "liar")).strip(),
                "speaker": str(row.get("speaker", "")).strip(),
                "party": str(row.get("party", "")).strip(),
                "subject": str(row.get("subject", "")).strip(),
                "job_title": str(row.get("job_title", "")).strip(),
                "state": str(row.get("state", "")).strip(),
                "context": str(row.get("context", "")).strip(),
                "label": label_text,
            }

            records.append({
                "id": metadata["id"],
                "claim": claim_text,
                "label": label_text,
                "metadata": metadata,
            })

        return records

    def _select_column(self, columns: Dict[str, str], candidates: Iterable[str]) -> Optional[str]:
        for candidate in candidates:
            if candidate in columns:
                return columns[candidate]
        return None
