import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

# Local application imports
from xai_compare.explainer_factory import ExplainerFactory
from xai_compare.config import EXPLAINERS


def run_and_collect_explanations(factory: ExplainerFactory, X_data, explainers=None, verbose=True) -> pd.DataFrame:
    results = []
    available_explainers = EXPLAINERS  # Easily extendable for additional explainers
    
    # chosen_explainers = explainers if explainers is not None else available_explainers

    if explainers is None:
        chosen_explainers = available_explainers
    elif type(explainers)==list:
        chosen_explainers = explainers
    else:
        chosen_explainers = [explainers]

    for explainer_type in chosen_explainers:
        explainer = factory.create_explainer(explainer_type)
        if explainer is not None:
            try:
                global_explanation = explainer.explain_global(X_data)
                results.append(global_explanation)
                if verbose:
                    print(f'\n {explainer_type.upper()} explanation created')
            except Exception as e:
                print(f"Failed to create {explainer_type.upper()} explanation: {e}")
        else:
            print(f"No explainer available for type: {explainer_type}")

    # Concatenate all results along columns (axis=1), handling cases where some explanations might fail
    if results:
        return pd.concat(results, axis=1)
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no explanations were added


def permutation_feature_importance(model, X_data, y_data, metric='accuracy', n_repeats=5, random_state=None):
    """
    Calculates permutation feature importance for a given model.
    
    Parameters:
    - model: The trained machine learning model.
    - X_data: The feature matrix.
    - y_data: The target vector.
    - metric: The metric to use for evaluating the model. Either 'accuracy' or 'mse'.
    - n_repeats: The number of times to permute a feature.
    - random_state: The random seed for reproducibility.
    
    Returns:
    - feature_importances_df: A DataFrame containing the feature names and their importance scores, sorted by scores.
    """
    # Set random seed for reproducibility if specified
    if random_state is not None:
        np.random.seed(random_state)
    
    if metric == 'accuracy':
        baseline_score = accuracy_score(y_data, model.predict(X_data))
        scorer = accuracy_score
    elif metric == 'mse':
        baseline_score = mean_squared_error(y_data, model.predict(X_data))
        scorer = mean_squared_error
    else:
        raise ValueError("Invalid metric. Please choose 'accuracy' or 'mse'.")
    
    # Initialize a dictionary to store importance scores for each feature
    feature_importances = {feature: [] for feature in X_data.columns}
    
    # Loop over each feature to permute and evaluate its importance
    for feature in X_data.columns:
        for _ in range(n_repeats):
            # Permute the current feature
            X_data_permuted = X_data.copy()
            X_data_permuted[feature] = np.random.permutation(X_data[feature])
            # Evaluate the model with the permuted feature and calculate the importance score
            permuted_score = scorer(y_data, model.predict(X_data_permuted))
            feature_importances[feature].append(baseline_score - permuted_score)

    # Average the importance scores over the number of repeats
    averaged_importances = {feature: np.mean(importances) for feature, importances in feature_importances.items()}
    
    # Normalize the importance scores to sum to 1 (for comparability with other explainers)
    total_importance = sum(averaged_importances.values())
    normalized_importances = {feature: importance / total_importance for feature, importance in averaged_importances.items()}
    
    # Convert to DataFrame 
    feature_importances_df = pd.DataFrame.from_dict(normalized_importances, orient='index', columns=['importance'])
    feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)
    
    return feature_importances_df
