"""
ML Predictor
Uses the pre-trained model to make predictions on news articles
"""

from typing import Dict, Tuple
from logger import logger_ml
from ml.model_loader import model_loader
from constants import LABEL_MAP, LABEL_FAKE, LABEL_REAL


class NewsArticlePredictor:
    """Wrapper for making predictions on news articles"""

    def __init__(self):
        """Initialize predictor with loaded model"""
        self.model = model_loader.get_model()
        logger_ml.info("NewsArticlePredictor initialized")

    def predict(self, text: str) -> Dict[str, any]:
        """
        Predict if article is fake or real

        Args:
            text: Article text to predict

        Returns:
            Dictionary with:
            - label: 0 (Fake) or 1 (Real)
            - label_text: "FAKE" or "REAL"
            - confidence: Float between 0 and 1
            - decision_score: Raw decision function score

        Raises:
            ValueError: If text is empty or invalid
            Exception: If prediction fails
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        if len(text.strip()) < 10:
            raise ValueError("Text must be at least 10 characters long")

        try:
            # Get prediction and decision score
            prediction = self.model.predict([text])[0]
            decision_score = self.model.decision_function([text])[0]

            # Convert decision score to confidence (0-1)
            # Using sigmoid function: 1 / (1 + exp(-score))
            import numpy as np

            confidence = 1 / (1 + np.exp(-abs(decision_score)))

            result = {
                "label": int(prediction),
                "label_text": LABEL_MAP.get(int(prediction), "UNKNOWN"),
                "confidence": float(confidence),
                "decision_score": float(decision_score),
                "is_real": bool(prediction == LABEL_REAL),
                "is_fake": bool(prediction == LABEL_FAKE),
            }

            logger_ml.debug(f"Prediction result: {result}")
            return result

        except Exception as e:
            logger_ml.error(f"❌ Prediction failed: {str(e)}")
            raise

    def batch_predict(self, texts: list) -> list:
        """
        Make predictions on multiple texts

        Args:
            texts: List of article texts

        Returns:
            List of prediction dictionaries
        """
        if not texts or not isinstance(texts, list):
            raise ValueError("Texts must be a non-empty list")

        try:
            predictions = []
            for text in texts:
                try:
                    pred = self.predict(text)
                    predictions.append(pred)
                except Exception as e:
                    logger_ml.warning(f"Failed to predict text: {str(e)}")
                    predictions.append({"error": str(e)})

            logger_ml.info(f"Batch prediction completed for {len(texts)} texts")
            return predictions

        except Exception as e:
            logger_ml.error(f"❌ Batch prediction failed: {str(e)}")
            raise

    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        try:
            model_info = {
                "model_type": type(self.model).__name__,
                "model_steps": list(self.model.named_steps.keys())
                if hasattr(self.model, "named_steps")
                else [],
                "is_fitted": hasattr(self.model, "n_features_in_"),
            }
            return model_info
        except Exception as e:
            logger_ml.error(f"Failed to get model info: {str(e)}")
            return {}


# Global predictor instance
predictor = NewsArticlePredictor()
