# Proof of Concept: ChemBERTaDNN Model

This directory contains a proof of concept implementation of a ChemBERTaDNN model for predicting molecular properties. 
The implementation includes data preprocessing, model training, and prediction scripts.
The model leverages the ChemBERTa architecture for molecular representation, with a DNN (currently with a single hidden 
layer of 32 units). ChemBERTa weights are finetuned during training.

## Directory Structure
 - train.py: Script to train the ChemBERTaDNN model.
 - predict.py: Script to make predictions using the trained model.
 - data/test_data.csv: Sample test dataset for predict.py
 - saved_models/: Directory to save trained models.
 - utils/: Utility functions for data processing and model handling.
 - model.py: Definition of the ChemBERTaDNN model architecture.
 - requirements.txt: List of required Python packages. # change this to pixi!!!!

## Setup
Install `pixi` (if not already installed):

```bash
      curl -fsSL https://pixi.sh/install.sh | bash
      pixi install
```

## To train 
To train the model run the training script. 
```bash
    python train.py --target ddr1 --split_index 1 --split_type random --save_dir saved_models
```
Optional arguments:
   --use_log --> if you want to use log-transformed y values. As default, it is False. 

## To predict
To predict using the trained model, run the prediction script:

```bash
    python predict.py --model_path saved_models/chemberta_model_split1.pth --input_data data/test_data.csv --output_path predictions.csv
```
The data to be predicted should be in a CSV file with a column named 'smiles' containing the SMILES strings.


