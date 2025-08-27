"""
Script to train the POC model.
Usage:
    python train.py --target ddr1 --split_index 1 --split_type random --save_dir saved_models
    python train.py --use_log --> if you want to use log-transformed y values

"""
import os
import time
import json
import logging
import argparse
import numpy as np
from collections import namedtuple
from utils import set_seed, save_chemberta_model, get_chemberta_model, get_training_data, get_testing_data
from evaluator import Evaluator
from model import ChemBERTaDNNWrapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train ChemBERTaDNN model.")
    parser.add_argument("--target", type=str, default="ddr1", help="Target dataset (e.g., ddr1, mapk14).")
    parser.add_argument("--split_index", type=int, default=1, help="Split index for data splitting.")
    parser.add_argument("--split_type", type=str, default="random", help="Split type (e.g., random, scaffold).")
    parser.add_argument("--save_dir", type=str, default="saved_models", help="Directory to save the model.")
    parser.add_argument("--use_log", action="store_true", help="Use log-transformed y values.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Set random seed for reproducibility
    set_seed(42)

    # Configuration
    chemberta_hyperparams = {
        "dnn_layers": [32],
        "dropout": 0.3,
        "lr": 5e-4,
        "epochs": 200,
        "patience": 5,
        "lr_scheduler": True,
        "lr_factor": 0.5,
        "lr_patience": 3,
        "freeze_chemberta": False,
        "batch_size": 64,
    }

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    df_train, df_valid, df_test = get_training_data(args.target, split_index=args.split_index, split_type=args.split_type)
    testing_data_heldout = get_testing_data(args.target)
    testing_data_inlib = get_testing_data(args.target, in_library=True)

    # Preprocess y values
    logger.info("Preprocessing y values...")
    if args.use_log:
        logger.info("Using log-transformed y values.")
        y_transform = np.log1p
    else:
        logger.info("Using original y values.")
        y_transform = lambda x: x

    X_train, y_train = df_train["smiles"].values, y_transform(df_train["y"].values)
    X_valid, y_valid = df_valid["smiles"].values, y_transform(df_valid["y"].values)
    X_test, y_test = df_test["smiles"].values, y_transform(df_test["y"].values)
    X_heldout_on, y_heldout_on = testing_data_heldout["on"]["smiles"].values, y_transform(testing_data_heldout["on"]["y"].values)
    X_heldout_off, y_heldout_off = testing_data_heldout["off"]["smiles"].values, y_transform(testing_data_heldout["off"]["y"].values)
    X_inlib_on, y_inlib_on = testing_data_inlib["on"]["smiles"].values, y_transform(testing_data_inlib["on"]["y"].values)
    X_inlib_off, y_inlib_off = testing_data_inlib["off"]["smiles"].values, y_transform(testing_data_inlib["off"]["y"].values)

    # Organize data
    AllDatasets = namedtuple("AllDatasets", ["train", "valid", "test", "heldout_on", "heldout_off", "inlib_on", "inlib_off"])
    Example = namedtuple("Example", ["x", "y", "y_original"])
    data = AllDatasets(
        train=Example(x=X_train.tolist(), y=y_train, y_original=df_train["y"].values),
        valid=Example(x=X_valid.tolist(), y=y_valid, y_original=df_valid["y"].values),
        test=Example(x=X_test.tolist(), y=y_test, y_original=df_test["y"].values),
        heldout_on=Example(x=X_heldout_on.tolist(), y=y_heldout_on, y_original=testing_data_heldout["on"]["y"].values),
        heldout_off=Example(x=X_heldout_off.tolist(), y=y_heldout_off, y_original=testing_data_heldout["off"]["y"].values),
        inlib_on=Example(x=X_inlib_on.tolist(), y=y_inlib_on, y_original=testing_data_inlib["on"]["y"].values),
        inlib_off=Example(x=X_inlib_off.tolist(), y=y_inlib_off, y_original=testing_data_inlib["off"]["y"].values),
    )

    # Load ChemBERTa model and tokenizer
    logger.info("Loading ChemBERTa model and tokenizer...")
    chemberta_model, tokenizer = get_chemberta_model()

    # Initialize and train the model
    logger.info("Initializing and training the model...")
    model = ChemBERTaDNNWrapper(chemberta_model=chemberta_model, tokenizer=tokenizer, **chemberta_hyperparams)
    model.fit(data.train.x, data.train.y, X_valid=data.valid.x, y_valid=data.valid.y)

    # Save the model
    logger.info("Saving the model...")
    save_chemberta_model(model, "chemberta", args.split_index, args.save_dir)

    # Evaluate the model
    logger.info("Evaluating the model...")
    evaluator = Evaluator(model, data, use_log=args.use_log)
    results = evaluator.evaluate()

    # Save results
    results_file = os.path.join(args.save_dir, f"results_split{args.split_index}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved results to {results_file}")

    # Log the time taken
    elapsed_time = time.time() - start_time
    logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    start_time = time.time()
    main()
