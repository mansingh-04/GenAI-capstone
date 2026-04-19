"""
Configuration management for Intelligent News Credibility Analyzer
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ML_MODEL_PATH = PROJECT_ROOT / "milestone1" / "model" / "news_credibility_model.pkl"
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_PATH = DATA_DIR / "chroma_db"
FACT_CHECK_DATA_PATH = DATA_DIR / "fact_check"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
FACT_CHECK_DATA_PATH.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# LLM Configuration
LLM_TYPE = os.getenv("LLM_TYPE", "groq")  # Options: "huggingface", "ollama", "groq"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "microsoft/DialoGPT-medium")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# External Embedding Selection (Save RAM on Render Free Tier)
USE_EXTERNAL_EMBEDDINGS = os.getenv("USE_EXTERNAL_EMBEDDINGS", "True").lower() == "true"
HUGGINGFACE_INFERENCE_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{os.getenv('EMBEDDING_MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2')}"

# News API Configuration
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
NEWSAPI_BASE_URL = "https://newsapi.org/v2"

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ChromaDB Configuration
CHROMA_COLLECTION_NAME = "fact_checks"
CHROMA_DISTANCE_METRIC = "cosine"

# RAG Configuration
STATIC_RAG_TOP_K = 5  # Number of results to retrieve
DYNAMIC_RAG_TOP_K = 5
EVIDENCE_RELEVANCE_THRESHOLD = 0.3  # Cosine similarity threshold

# Session Configuration
SESSION_TIMEOUT = 3600  # 1 hour in seconds
MAX_SESSIONS = 100

# Model Configuration
ML_CONFIDENCE_THRESHOLD = 0.5

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_DEBUG = os.getenv("API_DEBUG", "False").lower() == "true"
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "")

# Frontend → Backend connection (used by Streamlit UI; set in .env for deployment)
UI_BACKEND_HOST = os.getenv("UI_BACKEND_HOST", "http://localhost:8000")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# News sources configuration
NEWS_SOURCES_INDIA = [
    "bbc-news",
    "the-times-of-india",
    "the-hindu",
    "india-today",
]

NEWS_SOURCES_INTERNATIONAL = [
    "bbc-news",
    "cnn",
    "reuters",
    "the-guardian",
    "associated-press",
]

# Scrapers configuration (for web scraping fallback)
SCRAPER_TIMEOUT = 10  # seconds
SCRAPER_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Validation
if not ML_MODEL_PATH.exists():
    raise FileNotFoundError(f"ML Model not found at {ML_MODEL_PATH}")

if LLM_TYPE == "huggingface" and not HUGGINGFACE_API_KEY:
    print(
        "⚠️  WARNING: HUGGINGFACE_API_KEY not set. "
        "Set it in .env file or export environment variable."
    )

if not NEWSAPI_KEY:
    print(
        "⚠️  WARNING: NEWSAPI_KEY not set. "
        "Dynamic RAG will fall back to web scraping only."
    )

print(f"✅ Configuration loaded from: {BASE_DIR}")
