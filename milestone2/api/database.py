import sqlite3
from pathlib import Path
import json
import uuid
from datetime import datetime

# Setup SQLite Database in the data folder
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "history.db"

# Ensure database directory and table exist on module load
def _ensure_db():
    """Create database and tables if they don't exist."""
    try:
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
    except Exception as e:
        print(f"⚠️ Warning: Database initialization failed: {e}")

# Initialize DB immediately on module import
_ensure_db()

def init_db():
    """Explicitly initialize the database (safe to call multiple times)."""
    _ensure_db()

def save_analysis(claim: str, verdict: str, confidence: float, full_output: dict) -> str:
    """Save analysis to database, ensuring table exists first."""
    _ensure_db()  # Ensure table exists before insert
    analysis_id = str(uuid.uuid4())
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO analysis_history (id, timestamp, claim, verdict, confidence, raw_output) VALUES (?, ?, ?, ?, ?, ?)",
            (analysis_id, datetime.now().isoformat(), claim, verdict, confidence, json.dumps(full_output))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Failed to save analysis: {e}")
        _ensure_db()  # Try again
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO analysis_history (id, timestamp, claim, verdict, confidence, raw_output) VALUES (?, ?, ?, ?, ?, ?)",
                (analysis_id, datetime.now().isoformat(), claim, verdict, confidence, json.dumps(full_output))
            )
            conn.commit()
            conn.close()
        except Exception as e2:
            print(f"❌ Failed to save analysis on retry: {e2}")
    return analysis_id

def save_feedback(analysis_id: str, label: str, explanation: str):
    """Save feedback to database, ensuring table exists first."""
    _ensure_db()  # Ensure table exists before update
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE analysis_history SET feedback_label = ?, feedback_explanation = ? WHERE id = ?",
            (label, explanation, analysis_id)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Failed to save feedback: {e}")
        _ensure_db()  # Try again
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE analysis_history SET feedback_label = ?, feedback_explanation = ? WHERE id = ?",
                (label, explanation, analysis_id)
            )
            conn.commit()
            conn.close()
        except Exception as e2:
            print(f"❌ Failed to save feedback on retry: {e2}")

def get_history(limit: int = 50):
    """Fetch history from database, ensuring table exists first."""
    _ensure_db()  # Ensure table exists before query
    try:
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
    except Exception as e:
        print(f"⚠️ Error fetching history: {e}")
        _ensure_db()  # Try to reinit database
        try:
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
        except Exception as e2:
            print(f"❌ Error fetching history on retry: {e2}")
            return []
