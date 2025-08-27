import time
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, kendalltau
import shap
import matplotlib.pyplot as plt
import logging
import pickle
from evaluator import Evaluator
import sys
from pathlib import Path
from kindel.utils.data import featurize
from kindel.utils.helpers import set_seed # sets seed random np torch cuda
from collections import namedtuple
from kindel.utils.data import (
    get_training_data,
    get_testing_data,
    kendall,
    spearman,
    rmse,
)
import torch
import os
from explainer import Explainer
from featurizer import (MACCSFeaturizer, MorganFeaturizer,
                        PhysChemFeaturizer, SubstructureCountFeaturizer,
                        ChemBERTaFeaturizer, SMILESFeaturizer)
from dnn import DNNWrapper, ChemBERTaDNNWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "CPU")

set_seed(123)

def setup_logging(log_file):
    # Clear any existing handlers to avoid duplicate logs
    logging.getLogger().handlers = []

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # 'w' to overwrite, 'a' to append
            logging.StreamHandler()
        ]
    )


def get_chemberta_model():
    from transformers import AutoModel, AutoTokenizer
    # Load the pretrained ChemBERTa model and tokenizer
    model_name = "DeepChem/ChemBERTa-77M-MLM"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Move the model to the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, tokenizer


def save_chemberta_model(model, model_type, split_index, save_dir):
    """Save a ChemBERTaDNNWrapper model using torch.save."""
    model_file = os.path.join(save_dir, f"{model_type}_model_split{split_index}.pth")

    # Save the model's state_dict and hyperparameters
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'hyperparameters': {
            'dnn_layers': model.dnn_layers,
            'dropout': model.dropout,
            'lr': model.lr,
            'batch_size': model.batch_size,
            'epochs': model.epochs,
            'optimizer': model.optimizer_name,
            'weight_decay': model.weight_decay,
            'patience': model.patience,
            'lr_scheduler': model.lr_scheduler_flag,
            'lr_factor': model.lr_factor,
            'lr_patience': model.lr_patience,
            'freeze_chemberta': model.freeze_chemberta,
        }
    }
    torch.save(checkpoint, model_file)
    logging.info(f"Saved model to {model_file}")

def save_model(model, model_type, split_index, save_dir):
    """Save a PyTorch model using torch.save."""
    model_file = os.path.join(save_dir, f"{model_type}_model_split{split_index}.pth")

    # Save the model's state_dict and hyperparameters
    checkpoint = {
        'model_state_dict': model.model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'hyperparameters': {
            'input_dim': model.input_dim,
            'layers': model.layers,
            'dropout': model.dropout,
            'lr': model.lr,
            'batch_size': model.batch_size,
            'epochs': model.epochs,
            'optimizer': model.optimizer_name,
            'weight_decay': model.weight_decay,
            'patience': model.patience,
            'lr_scheduler': model.lr_scheduler_flag,
            'lr_factor': model.lr_factor,
            'lr_patience': model.lr_patience,
        }
    }
    torch.save(checkpoint, model_file)
    logging.info(f"Saved model to {model_file}")

FEATURIZERS = {
    "morgan": MorganFeaturizer,
    "maccs": MACCSFeaturizer,
    "chemberta": ChemBERTaFeaturizer,
    "physchem": PhysChemFeaturizer,
    "substructure": SubstructureCountFeaturizer,
    # "pharmacophore": PharmacophoreFeaturizer,
    "SMILES":SMILESFeaturizer,
}

def get_featurizer(featurizer_name):
    """Return an instance of the featurizer class."""
    return FEATURIZERS[featurizer_name]()


# ADD DNN
MODEL_HYPERPARAMS = {
    "xgb": {
        "n_estimators": 100,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
    },
    "knn": {
        "n_neighbors": 5,
    },
    "rf": {
        "n_estimators": 100,
        "criterion": "squared_error",
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    },
    "dnn": {
        "layers": [128, 64, 32],
        "dropout": 0.2,
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 50,
        "optimizer": "adam",
        "weight_decay": 1e-5,      # L2 regularization
        "patience": 10,            # early stopping patience
        "lr_scheduler": True,      # enable LR scheduler
        "lr_factor": 0.5,          # factor to reduce LR
        "lr_patience": 5,          # LR scheduler patience
    },
    "chemberta_dnn": {
        "dnn_layers": [128, 64, 32],
        "dropout": 0.2,
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 50,
        "optimizer": "adam",
        "weight_decay": 1e-5,  # L2 regularization
        "patience": 10,  # early stopping patience
        "lr_scheduler": True,  # enable LR scheduler
        "lr_factor": 0.5,  # factor to reduce LR
        "lr_patience": 5,  # LR scheduler patience
        "freeze_chemberta": False,  # Freeze ChemBERTa layers
    },
}

def get_model(model_name, hyperparams=None, input_dim=None, chemberta_model=None, tokenizer=None):
    """Return an instance of the model class with specified hyperparameters."""
    if hyperparams is None:
        hyperparams = MODEL_HYPERPARAMS.get(model_name, {})

    if model_name == "xgb":
        return XGBRegressor(**hyperparams)
    elif model_name == "knn":
        return KNeighborsRegressor(**hyperparams)
    elif model_name == "rf":
        return RandomForestRegressor(**hyperparams)
    elif model_name == "dnn":
        return DNNWrapper(input_dim=input_dim, **hyperparams)
    elif model_name == "chemberta_dnn":
        return ChemBERTaDNNWrapper(chemberta_model=chemberta_model, tokenizer=tokenizer, **hyperparams)

    else:
        raise ValueError(f"Unknown model: {model_name}")

def preprocess_all_datasets(df_train, df_valid, df_test, testing_data_heldout, testing_data_inlib,
                            featurizer, scale=True, feat_sel=True):
    """Preprocess all datasets and return them in a structured container."""
    start_time = time.time()

    # # for testing
    # df_train = df_train.sample(n=10000, random_state=42).reset_index(drop=True)
    # df_valid = df_valid.sample(n=500, random_state=42).reset_index(drop=True)
    # df_test = df_test.sample(n=500, random_state=42).reset_index(drop=True)

    time_featurize = start_time
    if featurizer:
        # Featurize all datasets
        X_train, y_train = featurizer.featurize_df(df_train, smiles_col="smiles", label_col="y")
        X_valid, y_valid = featurizer.featurize_df(df_valid, smiles_col="smiles", label_col="y")
        X_test, y_test = featurizer.featurize_df(df_test, smiles_col="smiles", label_col="y")
        X_heldout_on, y_heldout_on = featurizer.featurize_df(testing_data_heldout["on"], smiles_col="smiles", label_col="y")
        X_heldout_off, y_heldout_off = featurizer.featurize_df(testing_data_heldout["off"], smiles_col="smiles", label_col="y")
        X_inlib_on, y_inlib_on = featurizer.featurize_df(testing_data_inlib["on"], smiles_col="smiles", label_col="y")
        X_inlib_off, y_inlib_off = featurizer.featurize_df(testing_data_inlib["off"], smiles_col="smiles", label_col="y")

        time_featurize = time.time() - start_time
        logging.info(f"Time to featurize: {time_featurize:.2f} seconds")
        # Log original number of features
        n_features = X_train.shape[1]
        logging.info(f"Original number of features: {n_features}")
        logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logging.info(f"X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}")
        logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        logging.info(f"X_heldout_on shape: {X_heldout_on.shape}, y_heldout_on shape: {y_heldout_on.shape}")
        logging.info(f"X_heldout_off shape: {X_heldout_off.shape}, y_heldout_off shape: {y_heldout_off.shape}")
        logging.info(f"X_inlib_on shape: {X_inlib_on.shape}, y_inlib_on shape: {y_inlib_on.shape}")
        logging.info(f"X_inlib_off shape: {X_inlib_off.shape}, y_inlib_off shape: {y_inlib_off.shape}")


    else:
        X_train, y_train = df_train["smiles"].values, df_train["y"].values
        X_valid, y_valid = df_valid["smiles"].values, df_valid["y"].values
        X_test, y_test = df_test["smiles"].values, df_test["y"].values
        X_heldout_on, y_heldout_on = testing_data_heldout["on"]["smiles"].values, testing_data_heldout["on"][
            "y"].values
        X_heldout_off, y_heldout_off = testing_data_heldout["off"]["smiles"].values, testing_data_heldout["off"][
            "y"].values
        X_inlib_on, y_inlib_on = testing_data_inlib["on"]["smiles"].values, testing_data_inlib["on"]["y"].values
        X_inlib_off, y_inlib_off = testing_data_inlib["off"]["smiles"].values, testing_data_inlib["off"]["y"].values

        # Organize data for ChemBERTaDNN
        AllDatasets = namedtuple("AllDatasets",
                                 ["train", "valid", "test", "heldout_on", "heldout_off", "inlib_on", "inlib_off"])
        Example = namedtuple("Example", ["x", "y"])
        all_datasets = AllDatasets(
            train=Example(x=X_train.tolist(), y=y_train),
            valid=Example(x=X_valid.tolist(), y=y_valid),
            test=Example(x=X_test.tolist(), y=y_test),
            heldout_on=Example(x=X_heldout_on.tolist(), y=y_heldout_on),
            heldout_off=Example(x=X_heldout_off.tolist(), y=y_heldout_off),
            inlib_on=Example(x=X_inlib_on.tolist(), y=y_inlib_on),
            inlib_off=Example(x=X_inlib_off.tolist(), y=y_inlib_off)
        )
        logging.info("no featurizer. Returning raw SMILES.")
        return all_datasets, None, None

    time_scale = time_featurize
    # Scale all datasets if required
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)
        X_heldout_on = scaler.transform(X_heldout_on)
        X_heldout_off = scaler.transform(X_heldout_off)
        X_inlib_on = scaler.transform(X_inlib_on)
        X_inlib_off = scaler.transform(X_inlib_off)
        logging.info(f"Scaler mean: {scaler.mean_}")
        logging.info(f"Scaler variance: {scaler.var_}")
        time_scale = time.time() - time_scale
        logging.info(f"Time to scale: {time_scale:.2f} seconds")

    # Feature selection if required
    time_feat_sel = time_scale
    if feat_sel:
        selector = VarianceThreshold(threshold=0.1)
        X_train = selector.fit_transform(X_train)
        X_valid = selector.transform(X_valid)
        X_test = selector.transform(X_test)
        X_heldout_on = selector.transform(X_heldout_on)
        X_heldout_off = selector.transform(X_heldout_off)
        X_inlib_on = selector.transform(X_inlib_on)
        X_inlib_off = selector.transform(X_inlib_off)
        n_selected_features = X_train.shape[1]
        logging.info(f"Number of features after variance thresholding: {n_selected_features}")

        time_feat_sel = time.time() - time_feat_sel
        logging.info(f"Time to feature select: {time_feat_sel:.2f} seconds")


    # Organize all datasets
    # Organize all datasets
    AllDatasets = namedtuple(
        "AllDatasets",
        [
            "train", "valid", "test",
            "heldout_on", "heldout_off",
            "inlib_on", "inlib_off"
        ]
    )

    Example = namedtuple("Example", ["x", "y"])

    all_datasets = AllDatasets(
        train=Example(x=X_train, y=y_train),
        valid=Example(x=X_valid, y=y_valid),
        test=Example(x=X_test, y=y_test),
        heldout_on=Example(x=X_heldout_on, y=y_heldout_on),
        heldout_off=Example(x=X_heldout_off, y=y_heldout_off),
        inlib_on=Example(x=X_inlib_on, y=y_inlib_on),
        inlib_off=Example(x=X_inlib_off, y=y_inlib_off)
    )

    # Return datasets, scaler, and selector
    if scale and feat_sel:
        return all_datasets, scaler, selector
    elif scale and not feat_sel:
        return all_datasets, scaler, None
    elif not scale and feat_sel:
        return all_datasets, None, selector
    else:
        return all_datasets, None, None


def run_pipeline(featurizer_name,split_type,
                 model_name,hyperparams,
                 scale,feat_sel,explain,
            save_dir):
    """Run the modeling pipeline with logging and timing."""
    start_time = time.time()
    if featurizer_name:
        featurizer = get_featurizer(featurizer_name)
    else:
        featurizer = None
    split_indexes = [1]
    results = {}

    # run pipeline
    for split_index in split_indexes:
        logging.info(f"Starting pipeline for split index {split_index}")
        start_index_time = time.time()

        # get splits
        df_train, df_valid, df_test = get_training_data(target, split_index=split_index, split_type=split_type)
        # Get held-out and in-library datasets
        testing_data_heldout = get_testing_data(target)
        testing_data_inlib = get_testing_data(target, in_library=True)

        data, scaler, selector = preprocess_all_datasets(
            df_train, df_valid, df_test,
            testing_data_heldout,testing_data_inlib,
            featurizer, scale=scale, feat_sel=feat_sel
        )

        # RUN MODEL --- CHECK DIFFERENCES BETWEEN XGB KNN DNN ...
        time_preprocess = time.time()
        if model_type == "dnn":
            input_dim = data.train.x.shape[1]  # Number of features

            model = get_model(model_type, input_dim=input_dim, hyperparams=hyperparams)
            X_train_tensor = torch.tensor(data.train.x, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(data.train.y, dtype=torch.float32).to(device)
            X_valid_tensor = torch.tensor(data.valid.x, dtype=torch.float32).to(device)
            y_valid_tensor = torch.tensor(data.valid.y, dtype=torch.float32).to(device)
            model.fit(
                X_train_tensor, y_train_tensor,
                X_valid=X_valid_tensor, y_valid=y_valid_tensor
            )
            save_model(model, model_type, split_index, save_dir)

        elif model_type == "chemberta_dnn":
            chemberta_model, tokenizer = get_chemberta_model()# Load pre-trained ChemBERTa model
            model = get_model(
                model_name,
                hyperparams=hyperparams,
                chemberta_model=chemberta_model,
                tokenizer=tokenizer
            )

            model.fit(data.train.x, data.train.y, X_valid=data.valid.x, y_valid=data.valid.y)
            save_chemberta_model(model, model_type, split_index, save_dir)

        else:
            model = get_model(model_type, hyperparams=hyperparams)
            model.fit(data.train.x, data.train.y)
            model_file = os.path.join(save_dir, f"{model_type}_model_split{split_index}.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            logging.info(f"Saved model to {model_file}")
        train_time = time.time() - time_preprocess
        logging.info(f"time to train: {train_time:.2f} seconds")

        # Evaluate the model
        evaluator = Evaluator(model, data)
        results = evaluator.evaluate()

        # Save results
        results_file = os.path.join(save_dir, f"results_split{split_index}.json")

        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
        logging.info(f"Saved results to {results_file}")

        # SHAP analusis
        shap_time = time.time()
        if explain and model_name in ["xgb", "knn", "rf"]:
            shap_dir = os.path.join(save_dir, f"shap_split{split_index}")
            os.makedirs(shap_dir, exist_ok=True)
            explainer = Explainer(model, data.train.x, model_name, shap_dir, model_type=model_type)
            explainer.compute_shap()
            logging.info(f"time to compute SHAP values and plot: {time.time() - shap_time:.2f} seconds")

        total_time_i = time.time() - start_index_time
        logging.info(f"Total time for split index {split_index}: {total_time_i:.2f} seconds")
    # Log the time taken
    elapsed_time = time.time() - start_time
    logging.info(f"Pipeline completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":

    # split_indexes = [1, 2, 3, 4, 5]
    # featurizer = 'morgan'  # Options: 'morgan', 'maccs', 'chemberta', 'physchem', 'substructure'
    # model_type = 'xgb'  # Options: 'xgb', 'knn'
    # target = 'ddr1' # options ddr1, mapk14
    # split_type = 'random' # Options: "random", "disynthon"

    dnn_hyperparams = {
        "layers": [512, 256, 128, 64],
        "dropout": 0.3,
        "lr": 5e-4,
        "epochs": 500,
        "patience": 30,  # early stopping patience
        "lr_scheduler": True,  # enable LR scheduler
        "lr_factor": 0.5,  # factor to reduce LR
        "lr_patience": 10,  # LR scheduler patience
        }

    chemberta_hyperparams = {
        "dnn_layers": [32], #[512, 256, 128, 64],
        "dropout": 0.3,
        "lr": 5e-4,
        "epochs": 200,
        "patience": 5,  # early stopping patience
        "lr_scheduler": True,  # enable LR scheduler
        "lr_factor": 0.5,  # factor to reduce LR
        "lr_patience": 3,  # LR scheduler patience
        "freeze_chemberta":False,  # Allow fine-tuning
        "batch_size":64,  # 32, # 64 gave memory error

    }


    RUNS = [
        # # Format: (target, split_type, featurizer, scale, feat_sel, model_type, hyperparams, explain)
        ("ddr1", "random", "substructure", True, False, "xgb", {"n_estimators": 100}, True),
        ("ddr1", "random", "substructure", True, False, "knn", {"n_neighbors": 5}, False),
        ("ddr1", "random", "physchem", True, False, "xgb", {"n_estimators": 100}, True),
        ("ddr1", "random", "physchem", True, False, "knn", {"n_neighbors": 5}, False),
        ("ddr1", "random", "chemberta", False, False, "xgb", {"n_estimators": 100}, False),
        ("ddr1", "random", "chemberta", False, False, "knn", {"n_neighbors": 5}, False),
        ("ddr1", "random", "maccs", True, False, "xgb", {"n_estimators": 100}, True),
        ("ddr1", "random", "maccs", True, False, "knn", {"n_neighbors": 5}, False),
        # ("ddr1", "random", "morgan", True, True, "xgb", {"n_estimators": 100}, False),
        # # ("ddr1", "random", "morgan", True, True, "knn", {"n_neighbors": 5}, False),
        ("ddr1", "random", "substructure", True, False, "dnn", dnn_hyperparams, True),
        ("ddr1", "random", "physchem", True, False, "dnn", dnn_hyperparams, True),
        ("ddr1", "random", "chemberta", False, False, "dnn", dnn_hyperparams, False),
        ("ddr1", "random", "maccs", True, False, "dnn", dnn_hyperparams, True),
        # ("ddr1", "random", "morgan", True, False, "dnn", dnn_hyperparams, False),
        # ChEMBERT fine tune
        ("ddr1", "random", None, False, False, "chemberta_dnn", chemberta_hyperparams, False),

        # MAPK14
        ("mapk14", "random", "substructure", True, False, "xgb", {"n_estimators": 100}, True),
        ("mapk14", "random", "substructure", True, False, "knn", {"n_neighbors": 5}, False),
        ("mapk14", "random", "physchem", True, False, "xgb", {"n_estimators": 100}, True),
        ("mapk14", "random", "physchem", True, False, "knn", {"n_neighbors": 5}, False),
        ("mapk14", "random", "chemberta", False, False, "xgb", {"n_estimators": 100}, False),
        ("mapk14", "random", "chemberta", False, False, "knn", {"n_neighbors": 5}, False),
        ("mapk14", "random", "maccs", True, False, "xgb", {"n_estimators": 100}, True),
        ("mapk14", "random", "maccs", True, False, "knn", {"n_neighbors": 5}, False),
        ("mapk14", "random", "substructure", True, True, "dnn", dnn_hyperparams, True),
        ("mapk14", "random", "physchem", True, True, "dnn", dnn_hyperparams, True),
        ("mapk14", "random", "chemberta", False, False, "dnn", dnn_hyperparams, False),
        ("mapk14", "random", "maccs", True, True, "dnn", dnn_hyperparams, False),
        # ("mapk14", "random", "morgan", True, True, "dnn", dnn_hyperparams, False),
        ("mapk14", "random", None, False, False, "chemberta_dnn", chemberta_hyperparams, False),
    ]

    for run in RUNS:
        target, split_type, featurizer, scale, feat_sel, model_type, hyperparams, explain = run
        # Create a unique save directory for this run
        save_dir = f'results/{target}/{featurizer}_{model_type}_{split_type}/'
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, 'log.txt')
        setup_logging(log_file)  # Call this before any logging
        logging.info(f"Running pipeline for: split_type={split_type}, featurizer={featurizer}, model_type={model_type}, hyperparams={hyperparams}")
        logging.info(f"Hyperparameters: {hyperparams}")

        # Run pipeline
        run_pipeline(
            featurizer_name=featurizer,
            split_type=split_type,
            model_name=model_type,
            hyperparams=hyperparams,
            scale=scale,
            feat_sel=feat_sel,
            explain = explain,
            save_dir=save_dir,
        )



