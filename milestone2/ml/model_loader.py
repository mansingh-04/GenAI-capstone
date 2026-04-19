"""
ML Model Loader
Loads and caches the pre-trained TF-IDF + Linear SVM model from Milestone 1
"""

import joblib
from pathlib import Path
from logger import logger_ml
from config import ML_MODEL_PATH


class MLModelLoader:
    """Singleton loader for ML model"""

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLModelLoader, cls).__new__(cls)
        return cls._instance

    def load_model(self):
        """
        Load ML model from disk (cached in memory after first load)

        Returns:
            Trained pipeline object

        Raises:
            FileNotFoundError: If model file not found
            Exception: If model loading fails
        """
        if self._model is not None:
            logger_ml.debug("Returning cached ML model")
            return self._model

        try:
            logger_ml.info(f"Loading ML model from: {ML_MODEL_PATH}")

            if not ML_MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"ML Model not found at {ML_MODEL_PATH}. "
                    f"Please run Milestone 1 training first."
                )

            self._model = joblib.load(ML_MODEL_PATH)
            logger_ml.info("✅ ML model loaded successfully")
            return self._model

        except Exception as e:
            logger_ml.error(f"❌ Failed to load ML model: {str(e)}")
            raise

    def get_model(self):
        """Get model (loads if not already loaded)"""
        if self._model is None:
            self.load_model()
        return self._model

    def reload_model(self):
        """Force reload model from disk"""
        self._model = None
        logger_ml.info("Model cache cleared. Reloading on next call.")
        return self.load_model()


# Global instance
model_loader = MLModelLoader()
