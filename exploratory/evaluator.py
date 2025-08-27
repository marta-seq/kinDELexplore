import time
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mean_squared_error
import numpy as np
import logging
logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, model, all_datasets):
        """
        Initialize the evaluator with a trained model and preprocessed datasets.

        Args:
            model: Trained model (e.g., XGBoost, KNN, DNN).
            all_datasets: Preprocessed datasets (train, valid, test, heldout_on, heldout_off, inlib_on, inlib_off).
        """
        self.model = model
        self.all_datasets = all_datasets

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
            set_name="test"
        )

        # Held-out set ("on" and "off" conditions)
        results["all"] = {}
        for condition in ["on", "off"]:
            logging.info(f"Evaluating on held-out set ({condition})...")
            dataset = getattr(self.all_datasets, f"heldout_{condition}")
            results["all"][condition] = self._evaluate_set(
                dataset.x,
                dataset.y,
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
                set_name=f"inlib_{condition}"
            )

        return results

    def _evaluate_set(self, X, y, set_name):
        """
        Evaluate the model on a single dataset (X, y).
        Returns a dictionary of metrics (rho, tau, rmse).
        """
        start_time = time.time()
        logger.info(f"Predicting for {set_name}...")

        preds = self.model.predict(X)

        # Calculate metrics using scipy.stats
        rho, _ = spearmanr(preds, y)
        tau, _ = kendalltau(preds, y)
        rmse_val = np.sqrt(mean_squared_error(y, preds))

        logger.info(
            f"Results for {set_name}: "
            f"Spearman rho = {rho:.4f}, "
            f"Kendall tau = {tau:.4f}, "
            f"RMSE = {rmse_val:.4f}"
        )
        logger.info(f"Time to evaluate {set_name}: {time.time() - start_time:.2f} seconds")

        return {
            "rho": rho,
            "tau": tau,
            "rmse": rmse_val
        }

