# Emotional-Model

This repository hosts a dataset from X tailored for training a machine learning model to analyze emotions in user posts. The dataset is carefully annotated, and the model aims to provide insights into the emotional nuances prevalent in content shared on X.

## Dataset

The dataset consists of user posts collected from platform X. Each post is annotated with an emotion label, such as 'happy', 'sad', 'angry', etc. The annotations were done by experts in the field, ensuring high-quality labels for our model training.

## Model

We use a Logistic Regression model for this task. This model was chosen for its interpretability and efficiency. We use TF-IDF for feature extraction from the text data. The model is trained to predict the emotion label of a given user post.

## Evaluation

The model's performance is evaluated using precision, recall, and F1-score, which are common metrics for classification tasks. We also use a confusion matrix to visualize the performance of our model.

## Hyperparameter Tuning

We use GridSearchCV for hyperparameter tuning. This helps us find the optimal parameters for our Logistic Regression model, improving its performance.

## Usage

To use this model, you can clone this repository and run the Python script. Make sure to install the necessary Python libraries listed in the requirements.txt file.

## Future Work

We plan to explore more sophisticated models and feature extraction methods to further improve the performance of our emotion analysis model. We also aim to expand our dataset to include more diverse user posts.