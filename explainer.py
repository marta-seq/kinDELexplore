import shap
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import pickle
import torch

class Explainer:
    def __init__(self, model, X_train, model_name, save_dir, model_type="sklearn"):
        """
        Initialize the explainer with a trained model, training data, model name, and save directory.

        Args:
            model: Trained model (e.g., XGBoost, RandomForest, DNN).
            X_train: Training features (used for SHAP background).
            model_name: Name of the model (e.g., "xgb", "rf", "dnn").
            save_dir: Directory to save SHAP results.
            model_type: Type of model ("sklearn", "tf", "torch").
        """
        self.model = model
        self.X_train = X_train
        self.model_name = model_name
        self.save_dir = save_dir
        self.model_type = model_type
        os.makedirs(save_dir, exist_ok=True)

    def compute_shap(self):
        """
        Compute SHAP values for the model.
        Returns SHAP values and saves them to disk.
        """
        logging.info(f"Computing SHAP values for {self.model_name}...")

        # Initialize SHAP explainer based on model type
        if self.model_name in ["xgb", "rf"]:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.X_train)

        elif self.model_name == "dnn":
            if self.model_type == "tf":
                # For TensorFlow/Keras models
                explainer = shap.DeepExplainer(self.model, self.X_train[:100])  # Use a subset for background
                shap_values = explainer.shap_values(self.X_train)
            elif self.model_type == "torch":
                # For PyTorch models
                def model_predict(x):
                    with torch.no_grad():
                        return self.model(torch.FloatTensor(x)).numpy()
                explainer = shap.KernelExplainer(model_predict, self.X_train[:100])  # Use a subset for background
                shap_values = explainer.shap_values(self.X_train)
            else:
                logging.warning(f"Unsupported model type for DNN: {self.model_type}. Skipping SHAP...")
                return None

        else:
            logging.warning(f"SHAP not supported for {self.model_name}. Skipping...")
            return None

        # Save SHAP values
        shap_values_file = os.path.join(self.save_dir, "shap_values.pkl")
        with open(shap_values_file, 'wb') as f:
            pickle.dump(shap_values, f)
        logging.info(f"Saved SHAP values to {shap_values_file}")

        # Plot and save SHAP summary
        plt.figure()
        shap.summary_plot(shap_values, self.X_train, show=False)
        shap_plot_file = os.path.join(self.save_dir, "shap_summary.png")
        plt.savefig(shap_plot_file)
        plt.close()
        logging.info(f"Saved SHAP summary plot to {shap_plot_file}")

        return shap_values
