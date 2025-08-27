"""
Evaluator for the ChemBERTaDNN model.
"""
import numpy as np
import logging
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, model, all_datasets, use_log=False):
        """
        Initialize the evaluator with a trained model and preprocessed datasets.
        Args:
            model: Trained model (e.g., ChemBERTaDNNWrapper).
            all_datasets: Preprocessed datasets (train, valid, test, heldout_on, heldout_off, inlib_on, inlib_off).
            use_log: Whether the model was trained on log-transformed y values.
        """
        self.model = model
        self.all_datasets = all_datasets
        self.use_log = use_log

    def evaluate(self):
        """
        Evaluate the model on all test sets (internal, held-out, in-library).
        Returns a dictionary of results for each set and condition.
        """
        results = {}

        # Internal test set
        logger.info("Evaluating on internal test set...")
        results["test"] = self._evaluate_set(
            self.all_datasets.test.x,
            self.all_datasets.test.y,
            self.all_datasets.test.y_original,
            set_name="test"
        )

        # Held-out set ("on" and "off" conditions)
        results["all"] = {}
        for condition in ["on", "off"]:
            logger.info(f"Evaluating on held-out set ({condition})...")
            dataset = getattr(self.all_datasets, f"heldout_{condition}")
            results["all"][condition] = self._evaluate_set(
                dataset.x,
                dataset.y,
                dataset.y_original,
                set_name=f"heldout_{condition}"
            )

        # In-library set ("on" and "off" conditions)
        results["lib"] = {}
        for condition in ["on", "off"]:
            logger.info(f"Evaluating on in-library set ({condition})...")
            dataset = getattr(self.all_datasets, f"inlib_{condition}")
            results["lib"][condition] = self._evaluate_set(
                dataset.x,
                dataset.y,
                dataset.y_original,
                set_name=f"inlib_{condition}"
            )

        return results

    def _evaluate_set(self, X, y, y_original, set_name):
        """
        Evaluate the model on a single dataset (X, y, y_original).
        Returns a dictionary of metrics (rho, tau, rmse).
        """
        # Make predictions
        preds = self.model.predict(X)

        # Invert predictions if log-transform was used
        if self.use_log:
            preds = np.expm1(preds)
            y = np.expm1(y)

        # Calculate metrics
        rho, _ = spearmanr(y_original, preds)
        tau, _ = kendalltau(y_original, preds)
        rmse_val = np.sqrt(mean_squared_error(y_original, preds))

        logger.info(
            f"Results for {set_name}: "
            f"Spearman rho = {rho:.4f}, "
            f"Kendall tau = {tau:.4f}, "
            f"RMSE = {rmse_val:.4f}"
        )

        return {
            "rho": rho,
            "tau": tau,
            "rmse": rmse_val
        }
