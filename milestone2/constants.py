"""
Constants for the application
"""

# Prediction labels
LABEL_FAKE = 0
LABEL_REAL = 1
LABEL_INCONCLUSIVE = 2

LABEL_MAP = {
    LABEL_FAKE: "FAKE",
    LABEL_REAL: "REAL",
    LABEL_INCONCLUSIVE: "INCONCLUSIVE",
}

# Confidence levels
CONFIDENCE_HIGH = "High"
CONFIDENCE_MODERATE = "Moderate"
CONFIDENCE_LOW = "Low"

# Risk factors
RISK_FACTORS = [
    "Sensationalist language",
    "Lack of credible sources",
    "Contradicts fact-checks",
    "Uses unverified claims",
    "Emotional manipulation",
    "Extreme opinions without evidence",
]

# Response status codes
STATUS_SUCCESS = "success"
STATUS_ERROR = "error"
STATUS_PENDING = "pending"

# Error messages
ERROR_INVALID_INPUT = "Invalid input. Please provide text or URL."
ERROR_EXTRACTION_FAILED = "Failed to extract content from URL."
ERROR_MODEL_LOAD_FAILED = "Failed to load ML model."
ERROR_RAG_QUERY_FAILED = "Failed to query RAG system."
ERROR_LLM_REQUEST_FAILED = "Failed to process LLM request."
ERROR_AGENT_EXECUTION_FAILED = "Failed to execute agent workflow."

# Success messages
MSG_ANALYSIS_COMPLETE = "Analysis completed successfully."
MSG_CHATBOT_RESPONSE = "Chatbot response generated."

# Verdict types
VERDICT_LIKELY_FAKE = "Likely Fake"
VERDICT_LIKELY_REAL = "Likely Real"
VERDICT_MIXED = "Mixed Evidence"
VERDICT_INCONCLUSIVE = "Inconclusive"

# Time constants
HOUR_IN_SECONDS = 3600
DAY_IN_SECONDS = 86400

# Data formats
SUPPORTED_INPUT_TYPES = ["text", "url"]
SUPPORTED_OUTPUT_FORMAT = "json"
