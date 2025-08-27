"""
Script to load a saved model and make predictions.
Usage:
    python predict.py --model_file saved_models/chemberta_model_split1.pth \
                       --test_data_file data/test_data.csv \
                       --output_file predictions/test_predictions.csv \
                       --use_log
"""
import os
import logging
import argparse
import numpy as np
import pandas as pd
from utils import get_chemberta_model, load_chemberta_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Make predictions using a trained ChemBERTaDNN model.")
    parser.add_argument("--model_file", type=str, required=True, help="Path to the saved model file.")
    parser.add_argument("--test_data_file", type=str, required=True, help="Path to the test data CSV file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the predictions CSV file.")
    parser.add_argument("--use_log", action="store_true", help="Use log-transformed predictions.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load test data
    logger.info(f"Loading test data from {args.test_data_file}...")
    test_data = pd.read_csv(args.test_data_file)
    X_test = test_data["smiles"].tolist()
    # Load model and make predictions
    logger.info("Loading ChemBERTa model and tokenizer...")
    chemberta_model, tokenizer = get_chemberta_model()

    logger.info(f"Loading model from {args.model_file}...")
    model = load_chemberta_model(args.model_file, chemberta_model, tokenizer)

    logger.info("Making predictions...")
    preds = model.predict(X_test)

    # Invert predictions if log-transform was used
    if args.use_log:
        logger.info("Inverting log-transformed predictions...")
        preds = np.expm1(preds)

    # Create a DataFrame with true labels and predictions
    results = pd.DataFrame({
        "prediction": preds
    })

    # Save to CSV
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    results.to_csv(args.output_file, index=False)
    logger.info(f"Saved predictions to {args.output_file}")

    # Print sample predictions
    logger.info("Sample predictions:")
    print(results.head())

if __name__ == "__main__":
    main()

# python predict.py \
#     --model_file saved_models/chemberta_model_split1.pth \
#     --test_data_file data/test_data.csv \
#     --output_file predictions/test_predictions.csv \
#     --use_log