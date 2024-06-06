import pandas as pd
import numpy as np

from explainer_comparison.ExplainerFactory import ExplainerFactory
from explainer_comparison.explainer_utilities import run_and_collect_explanations

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler


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


def evaluate_explainers(model, X_data, y_data, metric='mse', threshold=0.2, random_state=None, verbose=True, explainers=["shap", "lime", "ebm"]):
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

    # Initialize variables to store resuts
    res_list = []
    list_el_feats = []

    # Loop until the number of features is reduced to the desired threshold
    while len(columns) > remaining_features:
        # Get SHAP and LIME values
        explainer_factory = ExplainerFactory(current_model, X_train=current_X, y_train=current_y)
        # shap_lime_importance = run_and_collect_explanations(explainer_factory, current_X)
        current_importance = run_and_collect_explanations(explainer_factory, current_X, verbose=verbose, explainers=explainers)

        # Calculate permutation feature importance
        feature_importances_dict = permutation_feature_importance(current_model, current_X, current_y, metric, random_state)
        feature_importances_df = pd.DataFrame.from_dict(feature_importances_dict, orient='index', columns=['importance'])

        # Find the least important feature
        sorted_feature_importances = feature_importances_df.sort_values(by='importance', ascending=True, key=abs)
        least_important_feature = sorted_feature_importances.index[0]
        
        # Print progress and results
        i = n - len(columns) + 1
        if verbose:
            print(f'\n {i} features eliminated. Now the least_important_feature is ', least_important_feature)


        results = pd.concat([current_importance, feature_importances_df], axis=1)
        if list_el_feats != []:
            for f in list_el_feats:
                results.loc[f] = [0] * results.shape[1]

        res_list.append(results)

        list_el_feats.append(least_important_feature)

        # Drop the least important feature
        current_X = current_X.drop(columns=[least_important_feature])
        columns.remove(least_important_feature)

        # Retrain the model with the reduced feature set
        current_model = base_model
        current_model.fit(current_X, current_y)

    # return a list of DataFrames with feature importance for each iteration
    return res_list


def plot_feat_importance(df):

    # find and eliminate rows with 0
    zero_rows = (df == 0).all(axis=1)
    df = df[~zero_rows]
    print(f'The least_important_feature is', df['importance'].abs().idxmin())

    df= pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns, index=df.index)

    fig, ax = plt.subplots(figsize=(12, 6))

    df.plot(kind='bar', ax=ax)
    ax.set_ylabel('Value')
    ax.set_xlabel('Feature')
    ax.set_title('SHAP, LIME, and Importance Values for Features')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# def evaluate_mse(results):

#     # Initialize variables to store SHAP and LIME MSE values
#     shap_values_prev = None
#     lime_values_prev = None
#     shap_mse_values = []
#     lime_mse_values = []

#     for df in results:
#         # get row idx with 0s
#         zero_rows_idx = df[(df == 0).all(axis=1)].index.values

#         # Get SHAP and LIME values
#         shap_values = df['SHAP Value']
#         lime_values = df['LIME Value']

#         # Calculate MSE if this is not the first iteration
#         if shap_values_prev is not None:
#             shap_mse = mean_squared_error(shap_values_prev.drop(index=zero_rows_idx), shap_values.drop(index=zero_rows_idx))
#             lime_mse = mean_squared_error(lime_values_prev.drop(index=zero_rows_idx), lime_values.drop(index=zero_rows_idx))
#             shap_mse_values.append(shap_mse)
#             lime_mse_values.append(lime_mse)
        
#         # Update previous values for the next iteration
#         shap_values_prev = shap_values.copy()
#         lime_values_prev = lime_values.copy()

#     return shap_mse_values, lime_mse_values


def evaluate_mse(results, explainer_keys=None):
    """
    Calculate MSE for multiple explainers' feature importance values across sequences of DataFrames.
    
    Args:
    - results: List of pandas DataFrames containing feature importance values from various explainers.
    - explainer_keys: List of keys in the DataFrames corresponding to different explainers.
    
    Returns:
    - A dictionary of lists containing MSE values for each explainer.
    """

    if explainer_keys is None:
        explainer_keys = [key for key in results[0].columns.values if key != 'importance'] 
        
    # Initialize dictionary to store previous values and MSE lists for each explainer
    prev_values = {key: None for key in explainer_keys}
    mse_values = {key: [] for key in explainer_keys}

    for df in results:
        # Get row idx with 0s
        zero_rows_idx = df[(df == 0).all(axis=1)].index
        
        # Loop through each explainer key
        for key in explainer_keys:
            values = df[key]

            # Calculate MSE if this is not the first iteration
            if prev_values[key] is not None:
                mse = mean_squared_error(prev_values[key].drop(index=zero_rows_idx), 
                                         values.drop(index=zero_rows_idx))
                mse_values[key].append(mse)
            
            # Update previous values for the next iteration
            prev_values[key] = values.copy()

    return mse_values


def plot_mse(mse_values_dict):
    # Plot the MSE values
    plt.figure(figsize=(10, 6))
    for k, v in mse_values_dict.items():
        plt.plot(v, label=k.replace('VALUE', "MSE"))
    # plt.plot(shap_mse_values, label='SHAP MSE')
    # plt.plot(lime_mse_values, label='LIME MSE')
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.title('Stability of Explainers Values')
    plt.legend()
    plt.grid(True)
    plt.show()
