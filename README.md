# Kinase Inhibitor Activity Prediction
This repository contains the code and analysis for predicting kinase inhibitor activity using Kindel dataset (https://arxiv.org/pdf/2410.08938)

The project is structured in three parts: exploratory analysis, proof of concept (PoC), and deployment planning.

## Structure of the repository
1. exploratory/
    * Exploratory analysis of different featurization methods and models.
    * Only data from random split 1 is used.
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
* Leveraging pre-trained transformer embeddings can be a promising approach, as it consistently improves performance compared to traditional featurization methods.
* However, overall predictive performance remains limited. The results indicate that the models, including ChemBERTa+DNN, still struggle to generalize to unseen compounds, especially in off-DNA test sets.

* Limitations:
    * No hyperparameter tuning or extensive architecture search was performed.
    * Limited to a subset of the dataset (random split 1).

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

