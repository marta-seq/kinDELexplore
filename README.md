# Kinase Inhibitor Activity Prediction
This repository contains the code and analysis for predicting kinase inhibitor activity using Kindel dataset (https://arxiv.org/pdf/2410.08938)

The project is divided into three main parts: exploratory analysis, proof of concept (PoC), and deployment planning.

## Structure of the repository

1. exploratory/
Exploratory analysis of different featurization methods and models.
Only the DDR1 daa from random split 1 is used for the exploratory analysis. 
Splitting, evaluation and metrics as the same as the paper. 

This directory contains the code to:
   - featurizer: Code to featurize the molecules using different methods. 
   Besides the Morgan fingerprints, implemented in Kindel, it implements Physicochemical descriptors, MACCS keys, Substructure Count, and the freezed ChemBERTa embeddings. 
   - visualization_compound_descriptors_ipynb: Jupyter notebook to visualize the different compound descriptors. No descriptor seems to separate the Kd values.
   - explainer: Code to obtain shap values for the different featurization methods.
   - evaluator: Code to evaluate the different models using the same metrics as the paper. 
   - dnn: contains a DNN wrapper model architecture and a ChemBERTa wrapper model architecture, that attachs a DNN head to the ChemBERTa embeddings, which are finetuned.
   - run: code to run full pipelines of testing different featurization models, different models ( XGBOost, KNN, DNN and ChemBERTa) and different hyperparameters. It evaluates and saves the results
   - aggregate_results: code to aggregate the results from the different runs. 
    - Pixi.toml env with the dependencies used for the exploratory analysis.
From the exploratory analysis, the ChemBERTa model with a DNN head was selected for the PoC. This model uses ChemBERTa embeddings with a 32-layer DNN head and fine-tunes the ChemBERTa embeddings.
2. POC/
This directory contains the code to run the POC model, which is the ChemBERTa model with a DNN head.
It has files to train, and predict. It also has a Pixi.toml file with the dependencies used for the POC model.

3. deployment/
Contains information for deploying the ChemBERTaDNN model as a production-ready system.

## Discussion 
On results ... 
ChemBERTa model outperforms the other models in all the splits and datasets.
On the test dataset the results, for all the models, are worse than on the validation dataset.

The ChemBERTaDNN model shows promising results in predicting kinase inhibitor activity, particularly when fine-tuning the ChemBERTa embeddings.

The model's performance could be further improved by exploring different architectures, hyperparameters, and additional featurization methods.

## Next steps 
Better understanding of the datasets, the problem and the metrics.
Experiment with different architectures and hyperparameters. Specially, try using MOLFORMER 
(in alternative to ChemBERTa) and check Graphs Neural Networks (GNNs). 
Enhance deployment. 
Check with the remaining of the datasets. 


Discussion and Future Work: Provide a critical analysis of your results, interpreting the significance of your findings in relation to your objectives. Suggest potential improvements, alternative approaches, or future directions that could enhance the model or address any unresolved issues. This can be as a notebook, slidedeck, part of the README or any other format you see fit.

Github repo: Please prepare a GitHub repository that showcases your approach to developing a machine learning model for DEL-ligand data. The repository should include code, architecture diagram, documentation, and any necessary instructions to understand and reproduce your work.
WTF is architecture dagram? and documentation? isnt README enough? 


## Evaluation Hints

Besides the technical solution you build, we’re also interested in how you:

- Identify problems before jumping into solutions.
- Communicate your reasoning.
- Structure and style matter. Write your code as if someone else will build on it tomorrow. We value clean, modular code.
- You don’t have to start from scratch — there are already many pretrained BioLLMs that might save you time if they fit your approach.
- You aren't expected to spend money in running this challenge but properly research your alternatives when presenting a solution.
- If you had another week, what would you build? Tell us.

You’ll present and defend your solution in the next interview round.