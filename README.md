# Kinase Inhibitor Activity Prediction
This repository contains the code and analysis for predicting kinase inhibitor activity using Kindel dataset (https://arxiv.org/pdf/2410.08938)

The project is structured in three parts: exploratory analysis, proof of concept (PoC), and deployment planning.

## Structure of the repository
1. exploratory/
    * Exploratory analysis of different featurization methods and models.
    * Only DDR1 data from random split 1 is used.
    * Splitting, evaluation, and metrics follow the Kindel paper.
    * From this analysis, the ChemBERTa+DNN model (ChemBERTa embeddings + 32-unit DNN head, fine-tuned) was selected for the PoC.

2. poc/
   * Code for the PoC model (ChemBERTa+DNN).
   * Includes scripts for training and prediction.
   * pixi.toml defines the environment dependencies.
   * Alternatively Docker image can be used.

3. deployment/
   * Contains information for deploying the ChemBERTaDNN model as a production-ready system.


## Results and Discussion
* ChemBERTa+DNN outperformed all other models in the exploratory phase. Fine-tuning ChemBERTa embeddings gave the best performance overall.
* On test datasets, all models performed worse than on validation sets (highlighting generalization issues).
* The above conclusions are valid both for DDR1 and MAPK14 targets.

# TODO 
This means that levering pre trained trasnformers can get good results. 
There is still room for improvement, as the results are not stellar.

* Limitations:
    * No hyperparameter tuning or extensive architecture search was performed.
    * Limited to a subset of the dataset (DDR1, split 1).

* Opportunities for improvement:
  * Explore different architectures and hyperparameters.
  * Test alternative transformer models (e.g., MolFORMER) and Graph Neural Networks (GNNs).
  * Incorporate biological context (e.g., protein target information).
  * Evaluate across the full Kindel dataset.

## Next Steps

* Deepen understanding of dataset properties, evaluation metrics, and biology/chemistry of the targets.
* Experiment with more architectures and featurizations.
* Enhance deployment pipeline.
* Extend experiments beyond split 1.

