import pandas as pd
import numpy as np

from explainer_comparison.ExplainerFactory import ExplainerFactory
from explainer_comparison.explainer_utilities import run_and_collect_explanations
from explainer_comparison.config import MODE

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
def get_feature_elimination_results(list_explainers, model, X_train, y_train, X_val, y_val, \
                                    X_test, y_test, mode, threshold=0.2, random_state=None, verbose=True):
    """
    Evaluates different feature elimination strategies provided by the list of explainers on a specified model.
    
    Each explainer is used to assess the importance of features, and based on that, evaluate the model's performance 
    with progressively eliminated features.

    Parameters:
    - list_explainers (list): A list of explainer instances or identifiers used to evaluate feature importance.
    - model: The machine learning model to be evaluated.
    - X_train, y_train, X_val, y_val, X_test, y_test: Training, validation, and testing datasets.
    - mode (str): The mode of operation for the explainers (e.g., 'classification', 'regression').
    - threshold (float, optional): The threshold for feature importance below which features are considered for elimination. Defaults to 0.2.
    - random_state (int, optional): A seed value to ensure reproducibility. Defaults to None.
    - verbose (bool, optional): If True, prints additional information during the function's execution. Defaults to True.

    Returns:
    - results_dict (dict): A dictionary containing the results from each explainer.
    """


    results_dict = {}

    for explainer in list_explainers:
        results_dict[explainer] = evaluate_explainer(model, X_train, y_train, X_val, y_val, X_test, y_test,\
                                                                explainer, mode, threshold=threshold, \
                                                                    random_state=random_state, verbose=verbose)
    
    return results_dict

def evaluate_explainer(model, X_train, y_train, X_val, y_val, X_test, y_test, explainer, mode, threshold=0.2, random_state=None, verbose=True,): #metric='mse', 
    """
    Evaluates the performance of a machine learning model with progressively fewer features based on the importance
    determined by various explainer methods.

    This function iteratively eliminates the least important features as determined by the specified explainer
    until the number of features is reduced to the desired threshold.

    Parameters:
    - model: Trained model to be evaluated.
    - X_train, y_train: Training data and labels.
    - X_val, y_val: Validation data and labels.
    - X_test, y_test: Test data and labels.
    - explainer (str): The type of explainer to use for feature importance evaluation.
    - mode (str): The mode of the operation, typically 'classification' or 'regression'.
    - threshold (float): Proportion of features to retain based on their importance.
    - random_state (int, optional): Seed used by random number generators for reproducibility.
    - verbose (bool, optional): If True, prints detailed progress information.

    Returns:
    - A list containing a list of DataFrames with feature importances and model evaluation results.
    """
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
        current_model_results = evaluate_models(current_model, X_train, y_train, X_val, \
                                                y_val, X_test, y_test, mode)
        res_model_eval.append(current_model_results)

        if explainer == 'permutation':
            # # Calculate permutation feature importance
            current_importance_dict = permutation_feature_importance(current_model, X_train, y_train, metric=metric, random_state=random_state) # current_y, metric,
            current_importance = pd.DataFrame.from_dict(current_importance_dict, orient='index', columns=['Permutation Value'])

        # temporary solution
        elif explainer == 'permutation_new':
            # # Calculate permutation feature importance
            current_importance_dict = permutation_feature_importance_new(current_model, X_train, random_state=random_state) # current_y, metric,
            current_importance = pd.DataFrame.from_dict(current_importance_dict, orient='index', columns=['Permutation Value'])
        
        else:
            # Get explainer values
            explainer_factory = ExplainerFactory(current_model, X_train=X_train, y_train=y_train)
            current_importance = run_and_collect_explanations(explainer_factory, X_train, verbose=verbose, explainers=explainer)

        # Find the least important feature
        # - apply absolute value on the global explanation results to get the feature importance for each XAI method.
        feature_importances = abs(current_importance)

        # - sort the list to have the order of the least to the most important features.
        sorted_feature_importances = feature_importances.sort_values(by=current_importance.columns[0], ascending=True)
        least_important_feature = sorted_feature_importances.index[0]

        # Log progress
        if verbose:
            print(f'Iteration: {len(columns) - len(X_train.columns) + 1}, Removed: {least_important_feature}')


        # results = pd.concat([current_importance, feature_importances_df], axis=1)
        results = sorted_feature_importances

        res_list.append(results) # add feature importance into the list

        list_el_feats.append(least_important_feature)

        # Drop the least important feature
        X_train = X_train.drop(columns=[least_important_feature])
        X_val = X_val.drop(columns=[least_important_feature])
        X_test = X_test.drop(columns=[least_important_feature])
        columns.remove(least_important_feature)

        # Retrain the model with the reduced feature set
        current_model = base_model
        current_model.fit(X_train, y_train)

    # return a list of DataFrames with model evaluation results
    return [res_list, res_model_eval]


# # - start an iterative process of eliminating the least important feature in each iteration 
# and store the evaluation on the train, validation, and test sets (train will be store for reporting) 
# - for classification store accuracy, precision, recall, auc. regression - MSE, MAE
def evaluate_models(model, X_train, y_train, X_val, y_val, X_test, y_test, mode):
    """
    Evaluates a model's performance metrics on training, validation, and test datasets.
    
    Parameters:
    - model: The model to evaluate.
    - X_train, y_train: Training data and labels.
    - X_val, y_val: Validation data and labels.
    - X_test, y_test: Test data and labels.
    - mode (str): Operation mode, 'classification' or 'regression'.

    Returns:
    - res_df (DataFrame): A DataFrame containing performance metrics for each dataset.

    The function calculates accuracy, precision, recall, and F1 scores for classification mode,
    and mean squared error and mean absolute error for regression mode.
    """

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

def add_best_feature_set(results_dict, mode, visualization=True):
    """
    Appends the best feature set analysis results to each entry in the results dictionary based on a specified metric.
    
    Parameters:
    - results_dict (dict): Dictionary containing results for different explainers or methods.
    - mode (str): Operating mode, which determines the main metric ('classification' for accuracy, 'regression' for MSE).
    - visualization (bool, optional): Whether to visualize the results during the process.
    
    Returns:
    - results_dict_upd (dict): Updated results dictionary with best feature set analysis appended.
    
    The function iterates over the results dictionary, applies a best feature set selection based on the specified
    main metric, and appends the results back into the dictionary.
    """

    if mode == MODE.CLASSIFICATION:
        main_metric = 'accuracy'
    else:
        main_metric = 'mse'

    results_dict_upd = results_dict.copy()

    for explnr,results in results_dict_upd.items():
        print('\033[1m' + explnr.upper() + '\033[0m')
        results_dict_upd[explnr].append(choose_best_feature_set(results[1], main_metric, visualization=visualization))
        print()

    return results_dict_upd


def choose_best_feature_set(model_ev_results, main_metric, data_type = 'val', visualization=True):
    """
    Evaluates and visualizes the best feature set based on a provided metric from model evaluation results.
    
    This function calculates the performance metrics for different numbers of features removed and identifies
    the optimal number of features by finding the highest metric value.

    Parameters:
    - model_ev_results (list): A list of DataFrame objects containing model evaluation metrics.
    - main_metric (str): The metric name to evaluate for the best feature set (e.g., 'accuracy', 'mse').
    - data_type (str, optional): Specifies the type of data ('train', 'val', or 'test') on which metrics are based.
    - visualization (bool, optional): If True, generates a plot to visualize the metrics across different feature sets.

    Returns:
    - num_eliminated_feats (int): The number of features suggested to be removed for optimal performance.
    """

    tdict = {}
    for metric in model_ev_results[0].index:
        metric_list = []
        for i in range(len(model_ev_results)):
            metric_list.append(model_ev_results[i].loc[[metric]][data_type].values[0])
        tdict[metric] = metric_list
    num_eliminated_feats = np.argmax(tdict[main_metric])

    if visualization:
        fig, ax = plt.subplots(figsize=(8, 5))

        tdf = pd.DataFrame(tdict)

        tdf.plot(ax=ax)

        plt.axvline(x=num_eliminated_feats, color='r', linestyle='--', label='Best feature set')

        # Set y-axis limits
        ax.set_ylim(0, 1)  # Set the limits of the y-axis to be from 0 to 1

        # Set axis labels
        ax.set_xlabel('Number of Features Eliminated')  
        ax.set_ylabel('Evaluation Metric Value')      

        # Set the plot title
        ax.set_title('Feature Elimination Analysis') 

        plt.show()

        print(f'{num_eliminated_feats} features are suggested to be removed')
        print(model_ev_results[num_eliminated_feats])

    return num_eliminated_feats  


# - after applying the process for each XAI method you should display the test score side by side
def plot_feature_selection_outcomes(results_dict_upd):
    """
    This function generate visualization of performance metrics for selected feature sets evaluated on the test set.

    Parameters:
    - results_dict_upd (dict): A dictionary where each key is an explainer name, and the value is
      a list containing a DataFrame of feature importances and model performance metrics for each feature set.

    This function creates two sets of plots: bar charts for feature importance and a table of metrics.
    """

    n_expl = len(results_dict_upd)

    # First subplot for bar plots
    fig, axs = plt.subplots(1, n_expl, figsize=(15, 6)) 

    for i, (explnr, results) in enumerate(results_dict_upd.items()):
        ax = axs if n_expl == 1 else axs[i]
        results[0][results[2]].plot(kind='barh', ax=ax, legend=False)
        ax.set_title(f'{explnr.upper()}\n feature importance\n Best result with \n{results[2]} features removed\n')

    fig.suptitle('Overall Feature Selection Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Second plot for the table
    fig, ax = plt.subplots(figsize=(15, 2)) 
    ax.axis('off')

    # get result without features elimination
    df_expl_results = round(list(results_dict_upd.values())[0][1][0]['test'], 4)
    for explnr, results in results_dict_upd.items():
        if df_expl_results is None:
            df_expl_results = round(results[1][results[2]]['test'], 4)
        else:
            df_expl_results = pd.concat([df_expl_results,(round(results[1][results[2]]['test'], 4))], axis=1)

    df_expl_results.columns = ['baseline_features_set'] + list(results_dict_upd.keys())

    pd.plotting.table(ax, df_expl_results, loc='upper center', colWidths=[0.15] * len(df_expl_results.columns))
    ax.set_title(f'Model Metrics for Selected Feature Sets on Test Set')

    plt.show()