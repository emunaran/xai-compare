import pandas as pd
import numpy as np

from explainer_comparison.ExplainerFactory import ExplainerFactory
from explainer_comparison.explainer_utilities import run_and_collect_explanations

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.base import clone


import matplotlib.pyplot as plt

    
def permutation_feature_importance(model, X_data, y_data, metric='accuracy', random_state=None):
    """
    Calculates permutation feature importance for a given model.
    
    Parameters:
    - model: The trained machine learning model.
    - X: The feature matrix.
    - y: The target vector.
    - metric: The metric to use for evaluating the model. Either 'accuracy' or 'mse'.
    - random_state: The random seed for reproducibility.
    
    Returns:
    - feature_importances_df: A DataFrame containing the feature names and their importance scores, sorted by scores.
    """
    if metric == 'accuracy':
        baseline_score = accuracy_score(y_data, model.predict(X_data))
        scorer = accuracy_score
    elif metric == 'mse':
        baseline_score = mean_squared_error(y_data, model.predict(X_data))
        scorer = mean_squared_error
    else:
        raise ValueError("Invalid metric. Please choose 'accuracy' or 'mse'.")
    
    feature_importances = {}
    for feature in X_data.columns:
        X_data_permuted = X_data.copy()
        X_data_permuted[feature] = np.random.permutation(X_data[feature])
        permuted_score = scorer(y_data, model.predict(X_data_permuted))
        feature_importances[feature] = baseline_score - permuted_score

    return feature_importances



def evaluate_explainers(model, X_data, y_data, metric='mse', threshold=0.2, random_state=None):
    # Copy the data to avoid modifying the original
    current_X = X_data.copy()
    current_y = y_data.copy()
    current_model = model

    # clone the model to get unfitted model
    base_model = clone(model)

    # Get the list of column names
    columns = X_data.columns.tolist()
    n = len(columns)
    remaining_features = n * threshold

    # Initialize variables to store SHAP and LIME MSE values
    shap_values_prev = None
    lime_values_prev = None
    shap_mse_values = []
    lime_mse_values = []

    # Loop until the number of features is reduced to the desired threshold
    while len(columns) > remaining_features:
        # Get SHAP and LIME values
        explainer_factory = ExplainerFactory(current_model, X_train=current_X, y_train=current_y)
        shap_lime_importance = run_and_collect_explanations(explainer_factory, current_X)
        shap_values = shap_lime_importance['SHAP Value']
        lime_values = shap_lime_importance['LIME Value']

        # Calculate MSE if this is not the first iteration
        if shap_values_prev is not None:
            shap_mse = mean_squared_error(shap_values_prev, shap_values)
            lime_mse = mean_squared_error(lime_values_prev, lime_values)
            shap_mse_values.append(shap_mse)
            lime_mse_values.append(lime_mse)

        # Update previous values for the next iteration
        shap_values_prev = shap_values.copy()
        lime_values_prev = lime_values.copy()

        # Calculate permutation feature importance
        feature_importances_dict = permutation_feature_importance(current_model, current_X, current_y, metric, random_state)
        feature_importances_df = pd.DataFrame.from_dict(feature_importances_dict, orient='index', columns=['importance'])

        # Find the least important feature
        sorted_feature_importances = feature_importances_df.sort_values(by='importance', ascending=True, key=abs)
        least_important_feature = sorted_feature_importances.index[0]

        # Print progress and results
        i = n - len(columns) + 1
        print(f'\n {i} features eliminated. Now the least_important_feature is ', least_important_feature)
        results = pd.concat([shap_lime_importance, feature_importances_df], axis=1)
        print(results)
        plot_results(results)

        # Drop the least important feature
        current_X = current_X.drop(columns=[least_important_feature])
        columns.remove(least_important_feature)

        # Remove the corresponding row from previous values
        shap_values_prev = shap_values_prev.drop(index=[least_important_feature])
        lime_values_prev = lime_values_prev.drop(index=[least_important_feature])

        # Retrain the model with the reduced feature set
        current_model = base_model
        current_model.fit(current_X, current_y)

    return shap_mse_values, lime_mse_values


def plot_results(df):

    max_importance = df['importance'].max()
    std_importance = df['importance'].std()

    # Apply z-score normalization to the 'importance' column
    df['importance'] = (df['importance'] - max_importance) / std_importance

    fig, ax = plt.subplots(figsize=(12, 6))

    df.plot(kind='bar', ax=ax)
    ax.set_ylabel('Value')
    ax.set_xlabel('Feature')
    ax.set_title('SHAP, LIME, and Importance Values for Features')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()