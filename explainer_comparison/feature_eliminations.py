import pandas as pd
import numpy as np

from explainer_comparison.ExplainerFactory import ExplainerFactory
from explainer_comparison.explainer_utilities import run_and_collect_explanations
from explainer_comparison.constants import MODE

from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score, roc_auc_score, mean_absolute_error, f1_score
from sklearn.base import clone
# from sklearn.preprocessing import StandardScaler

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
    
def permutation_feature_importance_new(model, X_data, random_state=None):
    """
    Calculates permutation feature importance for a given model.
    
    Parameters:
    - model: The trained machine learning model.
    - X: The feature matrix.
    - random_state: The random seed for reproducibility.
    
    Returns:
    - feature_importances: A dictionary containing the feature names and their mean importance.
    """

    # the importance by the permutation importance method goes as follows:
    # the more the intervened model decreases in its performance compared to the original model - the permutated feature is more important.

    # however, the explanations based on this method are as follows:
    # the local effect of each feature is measured by taking the differencing between the original local prediction to the intervened model local prediction:
    # original model predict(x) outcome minus intervened model predict(x).
    
    feature_importances = {}
    for feature in X_data.columns:
        X_data_permuted = X_data.copy()
        X_data_permuted[feature] = np.random.permutation(X_data[feature])
        feature_importances[feature] = np.mean(model.predict(X_data) - model.predict(X_data_permuted))
    return feature_importances


# - fit explainability method (each one separately).

# - for each explainability method (SHAP, LIME, Permutation, etc.) one should extract the global explanation in the 
# form of "feature impact" (aggregation of all local feature impact).

# - apply absolute value on the global explanation results to get the feature importance for each XAI method.

# - sort the list to have the order of the least to the most important features.

# - start an iterative process of eliminating the least important feature in each iteration 
# and store the evaluation on the train, validation, and test sets (train will be store for reporting) - 
# for classification store accuracy, precision, recall, auc. regression - MSE, MAE
def get_feature_elimination_results(list_explainers, model, X_train, y_train, X_val, y_val, X_test, y_test, mode, threshold=0.2, random_state=None, verbose=True):
    results_dict = {}

    for explainer in list_explainers:
        results_dict[explainer] = evaluate_explainers(model, X_train, y_train, X_val, y_val, X_test, y_test,\
                                                                explainer, mode, threshold=threshold, random_state=random_state, verbose=verbose)
    
    return results_dict

def evaluate_explainers(model, X_train, y_train, X_val, y_val, X_test, y_test, explainer, mode, threshold=0.2, random_state=None, verbose=True,): #metric='mse', 
    # Copy the data to avoid modifying the original
    current_X_train = X_train.copy()
    current_y_train = y_train.copy()
    current_X_val = X_val.copy()
    current_y_val = y_val.copy()
    current_X_test = X_test.copy()
    current_y_test = y_test.copy()
    current_model = model

    # clone the model to get unfitted model
    base_model = clone(model)

    if mode == MODE.CLASSIFICATION:
        metric = 'accuracy'
    else:
        metric = 'mse'

    # Get the list of column names
    columns = X_train.columns.tolist()
    n = len(columns)
    remaining_features = n * threshold

    # Initialize variables to store resuts
    res_list = [] # list of feature importance
    list_el_feats = [] # list of least important features

    res_model_eval = []

    # Loop until the number of features is reduced to the desired threshold
    while len(columns) > remaining_features:

        # evaluate the model
        current_model_results = evaluate_models(current_model, current_X_train, current_y_train, current_X_val, \
                                                current_y_val, current_X_test, current_y_test, mode)
        res_model_eval.append(current_model_results)

        if explainer == 'permutation':
            # # Calculate permutation feature importance
            current_importance_dict = permutation_feature_importance(current_model, current_X_train, current_y_train, metric=metric, random_state=random_state) # current_y, metric,
            current_importance = pd.DataFrame.from_dict(current_importance_dict, orient='index', columns=['Permutation Value'])

        # temporary solution
        elif explainer == 'permutation_new':
            # # Calculate permutation feature importance
            current_importance_dict = permutation_feature_importance_new(current_model, current_X_train, random_state=random_state) # current_y, metric,
            current_importance = pd.DataFrame.from_dict(current_importance_dict, orient='index', columns=['Permutation Value'])
        
        else:
            # Get explainer values
            explainer_factory = ExplainerFactory(current_model, X_train=current_X_train, y_train=current_y_train)
            current_importance = run_and_collect_explanations(explainer_factory, current_X_train, verbose=verbose, explainers=explainer)

        # Find the least important feature
        # - apply absolute value on the global explanation results to get the feature importance for each XAI method.
        feature_importances = abs(current_importance)

        # - sort the list to have the order of the least to the most important features.
        sorted_feature_importances = feature_importances.sort_values(by=current_importance.columns[0], ascending=True)
        least_important_feature = sorted_feature_importances.index[0]

        
        # Print progress and results
        i = n - len(columns) + 1
        if verbose:
            print(f'\n {i} features eliminated. Now the least_important_feature is ', least_important_feature)


        # results = pd.concat([current_importance, feature_importances_df], axis=1)
        results = sorted_feature_importances

        res_list.append(results) # add feature importance into the list

        list_el_feats.append(least_important_feature)

        # Drop the least important feature
        current_X_train = current_X_train.drop(columns=[least_important_feature])
        current_X_val = current_X_val.drop(columns=[least_important_feature])
        current_X_test = current_X_test.drop(columns=[least_important_feature])
        columns.remove(least_important_feature)

        # Retrain the model with the reduced feature set
        current_model = base_model
        current_model.fit(current_X_train, current_y_train)

    # return a list of DataFrames with model evaluation results
    return [res_list, res_model_eval]


# # - start an iterative process of eliminating the least important feature in each iteration 
# and store the evaluation on the train, validation, and test sets (train will be store for reporting) 
# - for classification store accuracy, precision, recall, auc. regression - MSE, MAE
def evaluate_models(model, X_train, y_train, X_val, y_val, X_test, y_test, mode):
    res_dict, res_df = {}, None
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    train_tpl = (y_train, y_pred_train, 'train')
    val_tpl = (y_val, y_pred_val, 'val')
    test_tpl = (y_test, y_pred_test, 'test')

    for y, y_pred, name in [train_tpl, val_tpl, test_tpl]:

        if mode == MODE.CLASSIFICATION:
            res_dict['accuracy'] = accuracy_score(y, y_pred)
            res_dict['precision'] = precision_score(y, y_pred)
            res_dict['recall'] = recall_score(y, y_pred)
            res_dict['f1'] = f1_score(y, y_pred)

            try:
                res_dict['auc'] = roc_auc_score(y, y_pred)
            except:
                pass
        
        else:
            res_dict['mse'] = mean_squared_error(y, y_pred)
            res_dict['mae'] = mean_absolute_error(y, y_pred)

        tmp_res_df = pd.DataFrame.from_dict(res_dict, orient='index', columns=[name])
        
        if res_df is None:
            res_df = tmp_res_df
        else:
            res_df = pd.concat([res_df, tmp_res_df], axis=1)

    return res_df

# - programmatically chose the best set of features based on a chosen evaluation metric (accuracy/ precision/ MSE...). 
# you can do that by applying argmax operation. iteration here = number of features to eliminate.

def add_best_feature_set(results_dict, mode):
    if mode == MODE.CLASSIFICATION:
        main_metric = 'accuracy'
    else:
        main_metric = 'mse'

    results_dict_upd = results_dict.copy()

    for explnr,results in results_dict_upd.items():
        print('\033[1m' + explnr.upper() + '\033[0m')
        results_dict_upd[explnr].append(choose_best_feature_set(results[1], main_metric))
        print()

    return results_dict_upd


def choose_best_feature_set(model_ev_results, main_metric, data_type = 'val'):
    tdict = {}
    for metric in model_ev_results[0].index:
        metric_list = []
        for i in range(len(model_ev_results)):
            metric_list.append(model_ev_results[i].loc[[metric]][data_type].values[0])
        tdict[metric] = metric_list
    num_eliminated_feats = np.argmax(tdict[main_metric])

    fig, ax = plt.subplots(figsize=(8, 5))

    tdf = pd.DataFrame(tdict)

    tdf.plot(ax=ax)

    # plt.axvline(x = num_eliminated_feats, color = 'r', linestyle='--')

    plt.axvline(x=num_eliminated_feats, color='r', linestyle='--', label='Best feature set')

    # Set y-axis limits
    ax.set_ylim(0, 1)  # Set the limits of the y-axis to be from 0 to 1

    # Set axis labels
    ax.set_xlabel('Number of Features Eliminated')  # X-axis label
    ax.set_ylabel('Evaluation Metric Value')       # Y-axis label

    # Set the plot title
    ax.set_title('Feature Elimination Analysis')  # Add a title to the plot

    plt.show()

    print(f'{num_eliminated_feats} features are suggested to be removed')
    print(model_ev_results[num_eliminated_feats])
    
    return num_eliminated_feats


# - after applying the process for each XAI method you should display the test score side by side
def plot_feat_select_results(results_dict_upd):
    n_expl = len(results_dict_upd)

    # Create a figure with 2 rows and n_expl columns
    fig, axs = plt.subplots(2, n_expl, figsize=(15, 6), height_ratios=[1, 2])  # Adjust the figure size as needed

    # First row for tables
    i = 0
    for explnr, results in results_dict_upd.items():
        ax = axs[0] if n_expl == 1 else axs[0, i]
        ax.axis('off')
        pd.plotting.table(ax, round(results[1][results[2]]['test'], 4), loc='upper right', colWidths=[.5, .5])
        ax.set_title(f'{explnr.upper()}\n{results[2]} features suggested \nto be removed')
        i += 1

    # Second row for bar plots
    i = 0
    for explnr, results in results_dict_upd.items():
        ax = axs[1] if n_expl == 1 else axs[1, i]
        results[0][results[2]].plot(kind='barh', ax=ax, legend=False)
        ax.set_title(f'Features importance \nby {explnr.upper()}')
        i += 1

    # Set the main title for the figure
    fig.suptitle('Overall Feature Selection Results', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.show()