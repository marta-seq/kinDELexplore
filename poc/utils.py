import os
import torch
import random
import numpy as np
import pandas as pd
from model import ChemBERTaDNNWrapper




DATA_ROOT = "s3://kin-del-2024/data"
url = "https://s3.42basepairs.com/kin-del-2024/data"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_training_data(target, split_index, split_type):
    # df = pd.read_parquet(os.path.join(DATA_ROOT, f"{target}_1M.parquet")).rename(
    #     {"target_enrichment": "y"}, axis="columns"
    # )
    # df = pd.read_parquet(os.path.join(url, f"{target}_1M.parquet"), engine="pyarrow")

    df = pd.read_parquet(
        f"s3://kin-del-2024/data/{target}_1M.parquet",
        storage_options={"anon": True}
    )

    df = df.rename({"target_enrichment": "y"}, axis="columns")

    # df_split = pd.read_parquet(
    #     os.path.join(DATA_ROOT, "splits", f"{target}_{split_type}.parquet")
    # )
    df_split = pd.read_parquet(
        f"s3://kin-del-2024/data/splits/{target}_{split_type}.parquet",
        storage_options={"anon": True}
    )
    return (
        df[df_split[f"split{split_index}"] == "train"],
        df[df_split[f"split{split_index}"] == "valid"],
        df[df_split[f"split{split_index}"] == "test"],
    )


def get_testing_data(target, in_library=False):
    data = {
        "on": pd.read_csv(
            os.path.join(DATA_ROOT, "heldout", f"{target}_ondna.csv"), index_col=0
        ).rename({"kd": "y"}, axis="columns"),
        "off": pd.read_csv(
            os.path.join(DATA_ROOT, "heldout", f"{target}_offdna.csv"), index_col=0
        ).rename({"kd": "y"}, axis="columns"),
    }



    if in_library:
        data["on"] = data["on"].dropna(subset="molecule_hash")
        data["off"] = data["off"].dropna(subset="molecule_hash")
    return data

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
def load_chemberta_model(model_file, chemberta_model, tokenizer):
    """Load a saved ChemBERTaDNNWrapper model."""
    checkpoint = torch.load(model_file)

    # Reinitialize the model with the same hyperparameters
    model = ChemBERTaDNNWrapper(
        chemberta_model=chemberta_model,
        tokenizer=tokenizer,
        dnn_layers=checkpoint['hyperparameters']['dnn_layers'],
        dropout=checkpoint['hyperparameters']['dropout'],
        lr=checkpoint['hyperparameters']['lr'],
        batch_size=checkpoint['hyperparameters']['batch_size'],
        epochs=checkpoint['hyperparameters']['epochs'],
        optimizer=checkpoint['hyperparameters']['optimizer'],
        weight_decay=checkpoint['hyperparameters']['weight_decay'],
        patience=checkpoint['hyperparameters']['patience'],
        lr_scheduler=checkpoint['hyperparameters']['lr_scheduler'],
        lr_factor=checkpoint['hyperparameters']['lr_factor'],
        lr_patience=checkpoint['hyperparameters']['lr_patience'],
        freeze_chemberta=checkpoint['hyperparameters']['freeze_chemberta'],
    )

    # Load the saved state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    return model
def save_chemberta_model(model, split_index, save_dir):
    """Save a ChemBERTaDNNWrapper model using torch.save."""
    model_file = os.path.join(save_dir, f"chemberta_model_split{split_index}.pth")

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
