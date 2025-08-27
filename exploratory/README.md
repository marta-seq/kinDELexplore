## Exploratory Analysis of Different Featurization Methods and Models

In this first stage, I explored different featurization methods and models for predicting kinase inhibitor activity, using random split 1 of the Kindel dataset.

The aim was to assess whether alternative featurizations (beyond the Morgan fingerprints used in the Kindel paper) and different models could yield better performance.

I tested:

* Models: XGBoost, KNN (as in the original paper), and DNNs.
* Featurizations: Morgan fingerprints, physicochemical descriptors, MACCS keys, substructure counts, and frozen ChemBERTa embeddings.
* ChemBERTa: evaluated both frozen embeddings and fine-tuned embeddings with a DNN head.

A visualization notebook was used to explore how descriptors separate Kd values. 

Not all combinations of features and models were tested due to time and compute constraints.

Splits, evaluation, and metrics followed the Kindel paper. Extending the evaluation to additional metrics and strategies would be beneficial.

## Structure of the Repository
This directory contains:

* featurizer: Code to featurize the molecules using different methods. 
   Besides the Morgan fingerprints, implemented in Kindel, it implements Physicochemical descriptors, MACCS keys, Substructure Count, and the freezed ChemBERTa embeddings. 
* visualization_compound_descriptors_ipynb: Jupyter notebook to visualize descriptors (none clearly separated Kd values).
* explainer: Code to obtain shap values for the different featurization methods.
* evaluator: Code to evaluate the different models using the same metrics as the Kindel paper. 
* dnn: DNN wrapper and ChemBERTa+DNN wrapper (with fine-tuning).
* run: code to run full pipelines of testing different featurization models, different models ( XGBOost, KNN, DNN and ChemBERTa) and different hyperparameters. It evaluates and saves the results
* aggregate_results: code to aggregate the results from the different runs. 
* results/all_results.csv: aggregated results from the different runs.
* pixi.toml - env with the dependencies used for the exploratory analysis.
From the exploratory analysis, the ChemBERTa model with a DNN head was selected for the PoC. This model uses ChemBERTa embeddings with a 32-layer DNN head and fine-tunes the ChemBERTa embeddings.

## Results
* XGBoost and KNN models were tested with different featurization methods and did not performed well.
* ChemBERTa+DNN gave the best results, particularly when fine-tuning embeddings.
* However, no model generalized well to the test set, especially the off-DNA set.
* None of the models outperformed the initial benchmark.

Limitations:

* No grid search of hyperparameters; only one DNN architecture tested.

* Results could be improved by exploring more architectures, hyperparameters, and featurizations.

* Other transformer models (e.g. MolFORMER) or GNNs could be tested.

* A deeper understanding of the datasets, DDR1 and MAPK14 biology, and leveraging protein target information could further improve results.

## Conclusion
From this analysis, the ChemBERTa+DNN model was selected for the PoC. It combines fine-tuned ChemBERTa embeddings with a 32-unit DNN head.