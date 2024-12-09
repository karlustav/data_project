# Exploring Mental Health Data

## Motivation for the Project
- Depression is an escalating global issue, impacting millions across diverse demographics. While awareness is on the rise, countless individuals at risk continue to go undiagnosed or untreated. This project explores the connections between everyday factors and mental health risks.

## Goal
- The primary goal of this project is to develop a machine learning-based predictive model that identifies individuals at risk of depression using demographic and lifestyle data.
- Since this was a Kaggle competition, we also tried to get a good place on the Leaderboard using deep learning.

## Repository Structure

### Data Processing (part of Miscellaneous)
- `prepare_and_explore/`
  - `init_data_exploration.ipynb`: Initial data analysis and visualization
  - `prepare_data.ipynb`: Training data preprocessing pipeline
  - `prepare_test_data.ipynb`: Test data preprocessing pipeline
  - `drop_all_not_important.ipynb`: Feature selection and dimensionality reduction

### Models
- `learn/`
  - `models.ipynb`: Traditional ML models (Random Forest, XGBoost, etc.)
  - `deeplearn/`
    - `tf_model_14.ipynb`: Final TensorFlow deep learning implementation
    - `tf_model_v{x}.ipynb`: Version x of TensorFlow deep learning implementation
    - `tf_model.ipynb`: Initial deep learning experiments
    - `tf_model_v2.py`: Python script version of the model

### Results and Analysis
- `results/`
  - `best_models.ipynb`: Comparison of model performances
  - `feature_importance.ipynb`: Analysis of feature importance

### Submissions
- `submissions/`: Contains various model prediction submissions
  - Default model submissions (DT, RF, SVM, XGBoost)
  - Optimized model submissions
  - Sample submission format

### Miscellaneous
- `misc/`: Backup implementations and utility files
  - Various test implementations
  - Configuration files
  - Requirements and dependencies

## Model Architecture
The final model (`tf_model_14.ipynb`) implements:
- Deep neural network with attention mechanism
- Residual connections
- Batch normalization
- Dropout regularization
- Custom learning rate scheduling
- Class weight balancing

## Setup and Usage

### Data Preparation
1. Place raw data in `data/` directory
2. Run preprocessing notebooks in `prepare_and_explore/`
3. Use processed data for model training

### Training
1. For traditional ML: Run `learn/models.ipynb`
2. For deep learning: Run `learn/deeplearn/tf_model_14.ipynb`

### Generating Predictions
The models will automatically save predictions to the `submissions/` directory.

## Results
Best performing models and their metrics can be found in `results/best_models.ipynb`.


## Team members
- Karl-Ustav Kõlar
- Oskar Pukk
