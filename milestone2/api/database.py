import sqlite3
from pathlib import Path
import json
import uuid
from datetime import datetime

# Setup SQLite Database in the data folder
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "history.db"

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create Analysis History Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS analysis_history (
        id TEXT PRIMARY KEY,
        timestamp TEXT,
        claim TEXT,
        verdict TEXT,
        confidence REAL,
        raw_output TEXT,
        feedback_label TEXT,
        feedback_explanation TEXT
    )
    ''')
    conn.commit()
    conn.close()

def save_analysis(claim: str, verdict: str, confidence: float, full_output: dict) -> str:
    analysis_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO analysis_history (id, timestamp, claim, verdict, confidence, raw_output) VALUES (?, ?, ?, ?, ?, ?)",
        (analysis_id, datetime.now().isoformat(), claim, verdict, confidence, json.dumps(full_output))
    )
    conn.commit()
    conn.close()
    return analysis_id

def save_feedback(analysis_id: str, label: str, explanation: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE analysis_history SET feedback_label = ?, feedback_explanation = ? WHERE id = ?",
        (label, explanation, analysis_id)
    )
    conn.commit()
    conn.close()

def get_history(limit: int = 50):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, timestamp, claim, verdict, confidence, feedback_label FROM analysis_history ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
