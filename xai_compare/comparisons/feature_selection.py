# ----------------------------------------------------------------------------------------------------
# Module Comparison
#
# This module contains classes for comparing model Explainers, evaluating their consistency, and assessing 
# feature selection strategies. It provides a framework for generating comparison reports, visualizing results, 
# and measuring the stability and performance of different explainer methods on machine learning models.
#
# ------------------------------------------------------------------------------------------------------

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Union, Type
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, mean_squared_error, precision_score, 
                    recall_score, roc_auc_score, mean_absolute_error, f1_score)
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Local application imports
from xai_compare.config import MODE, EXPLAINERS
from xai_compare.comparison import Comparison
from xai_compare.explainer import Explainer
from xai_compare.explainer_utilities import run_and_collect_explanations


class FeatureSelection(Comparison):

    """
    A class to evaluate different feature elimination strategies provided by the list of explainers on a specified model.

    Attributes:
    - model (Model): The machine learning model to be evaluated.
    - data: data
    - target: labels.
    - mode (str): The mode of operation for the explainers ('classification', 'regression').
    - threshold (float): The threshold for feature importance below which features are considered for elimination.
    - random_state (int): A seed value to ensure reproducibility.
    - verbose (bool): If True, prints additional information during the function's execution.
    """

    def __init__(self,
                 model,
                 data: pd.DataFrame,
                 target: Union[pd.DataFrame, pd.Series, np.ndarray],
                 mode: str = MODE.REGRESSION, 
                 random_state=42, 
                 verbose=True,
                 threshold=0.2,
                 metric=None, 
                 default_explainers=EXPLAINERS,
                 custom_explainer=None):
        
        if not hasattr(model, 'fit'):
            raise ValueError("The model should have a 'fit' method.")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data should be a pandas DataFrame.")
        if not isinstance(target, (pd.DataFrame, pd.Series, np.ndarray)):
            raise TypeError("Target should be a pandas DataFrame, Series, or a numpy ndarray.")
        if mode not in [MODE.REGRESSION, MODE.CLASSIFICATION]:
            raise ValueError(f"Invalid mode. Expected 'REGRESSION' or 'CLASSIFICATION', got {mode}.")
        if not isinstance(random_state, int):
            raise TypeError("Random state should be an integer.")
        if not isinstance(verbose, bool):
            raise TypeError("Verbose should be a boolean.")
        if not isinstance(default_explainers, list):
            raise TypeError("Default explainers should be a list of strings.")
        if not all(explainer in EXPLAINERS for explainer in default_explainers):
            raise ValueError(f"Some default explainers are not in the allowed EXPLAINERS list: {default_explainers}")
        if custom_explainer and not isinstance(custom_explainer, (Type[Explainer], list, None)):
            raise TypeError("Custom explainer should be an Explainer type, a list of Explainer types, or None.")
        if not isinstance(threshold, (int, float)) or not (0 < threshold <= 1):
            raise ValueError("Threshold should be a float between 0 and 1.")

        super().__init__(model, data, target, mode=mode, verbose=verbose, random_state=random_state, 
                         default_explainers=default_explainers, custom_explainer=custom_explainer)
        self.threshold = threshold
        self.results = {}
        self.df_expl_results = None
        self.results_dict_upd = None
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.train_test_val()

        if metric is None:
            if self.mode == MODE.CLASSIFICATION:
                self.metric = 'Accuracy'
            else:
                self.metric = 'MSE'
        else:
            # Validate the chosen metric
            if metric not in ['Accuracy', 'Precision', 'Recall', 'F1_score', 'AUC', 'MSE', 'MAE']:
                raise ValueError(f"Invalid metric '{metric}'. Valid options are 'Accuracy', 'Precision', 'Recall', 'F1_score', 'AUC', 'MSE', 'MAE'.")
            self.metric = metric

    def train_test_val(self):
        """
        Splits the data into training, validation, and testing sets.

        This method splits the data into a training set, a validation set, and a testing set.
        It uses a stratified split to maintain the distribution of the target variable across the sets.

        Returns:
        -------
        tuple:
            - X_train (pd.DataFrame): Training features.
            - y_train (Union[pd.DataFrame, pd.Series, np.ndarray]): Training target.
            - X_val (pd.DataFrame): Validation features.
            - y_val (Union[pd.DataFrame, pd.Series, np.ndarray]): Validation target.
            - X_test (pd.DataFrame): Testing features.
            - y_test (Union[pd.DataFrame, pd.Series, np.ndarray]): Testing target.
        """
        X_train, X_tmp, y_train, y_tmp = train_test_split(self.data, self.y, test_size=0.5, random_state=self.random_state)  
        X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.4, random_state=self.random_state)  
        return X_train, y_train, X_val, y_val, X_test, y_test

    def apply(self):
        if self.results_dict_upd is None:
            self.get_feature_elimination_results()
            self.add_best_feature_set()
        else:
            pass

    def display(self):
        self.plot_feature_selection_outcomes()

    def best_result(self):
        """
        Determines the best feature set based on the specified metric and optionally visualizes the results.

        Returns:
        -------
        The maximum value of the specified evaluation metric across all explainer results,
        indicating the best feature set performance.
        """
        if self.results_dict_upd:
            df_expl_results = round(list(self.results_dict_upd.values())[0][1][0]['test'], 4)
            for explnr, results in self.results_dict_upd.items():
                if df_expl_results is None:
                    df_expl_results = round(results[1][results[2]]['test'], 4)
                else:
                    df_expl_results = pd.concat([df_expl_results, (round(results[1][results[2]]['test'], 4))], axis=1)

            df_expl_results.columns = ['baseline_features_set'] + list(self.results_dict_upd.keys())
            self.df_expl_results = df_expl_results
        else:
            self.get_feature_elimination_results()
            self.add_best_feature_set()
            self.best_result()

        return self.df_expl_results.loc[[self.metric]].max(axis=1)

    def get_feature_elimination_results(self):
        """
        Evaluates different feature elimination strategies provided by the list of explainers on a specified model.
        
        Each explainer is used to assess the importance of features, and based on that, evaluate the model's performance 
        with progressively eliminated features.

        Returns:
        - results_dict (dict): A dictionary containing the results from each explainer.
        """
        results_dict = {}
        for explainer in tqdm(self.list_explainers, desc="Explainers"):
            results_dict[explainer.__name__] = self.evaluate_explainer(explainer)
        self.results_dict = results_dict

    def evaluate_explainer(self, explainer): 
        """
        Evaluates the performance of a machine learning model with progressively fewer features based on the importance
        determined by various explainer methods.

        This function iteratively eliminates the least important features as determined by the specified explainer
        until the number of features is reduced to the desired threshold.

        Returns:
        - A list containing a list of DataFrames with feature importances and model evaluation results.
        """
        X_train, X_val, X_test = self.X_train, self.X_val, self.X_test

        # Clone the model to get unfitted model
        unfitted_model = self._clone_model()

        # Get the list of column names
        columns = self.X_train.columns.tolist()
        n = len(columns)
        remaining_features = n * self.threshold

        # Initialize variables to store results
        res_list = [] # list of feature importance
        list_el_feats = [] # list of least important features

        res_model_eval = []

        current_model = unfitted_model

        # Loop until the number of features is reduced to the desired threshold
        while len(columns) > remaining_features:
            current_model.fit(X_train, self.y_train)

            # evaluate the model
            current_model_results = self.evaluate_models(current_model, X_train, self.y_train, X_val, \
                                                    self.y_val, X_test, self.y_test, self.mode)
            res_model_eval.append(current_model_results)
        
            # Get explainer values
            explainer_instance = copy.copy(explainer) 
            explainer_instance = explainer_instance(current_model, X_train, self.y_train)
            current_importance = run_and_collect_explanations(explainer_instance, X_train, verbose=self.verbose)

            # Find the least important feature
            # Apply absolute value on the global explanation results to get the feature importance for each XAI method.
            feature_importances = abs(current_importance)

            # Sort the list to have the order of the least to the most important features.
            sorted_feature_importances = feature_importances.sort_values(by=current_importance.columns[0], ascending=True)
            least_important_feature = sorted_feature_importances.index[0]

            # Log progress
            if self.verbose:
                print(f'Iteration: {n - len(X_train.columns) + 1}, Removed: {least_important_feature}')

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
            current_model = unfitted_model

        # return a list of DataFrames with model evaluation results
        return [res_list, res_model_eval]

    def _clone_model(self):
        """
        Clones a model based on its type.
        
        Returns:
        - cloned_model: The cloned model.
        """
        class XGBClassifierWrapper(xgb.XGBClassifier):
            def __init__(self, verbose=False, **kwargs):
                """
                Initialize the XGBClassifierWrapper with model parameters.
                """
                default_params = {}
                default_params.update(kwargs)
                super().__init__(**default_params)
                self.verbose = verbose

            def __call__(self, X):
                """
                Make the model callable to be compatible with SHAP.
                """
                return self.predict_proba(X)

        # clone the model to get unfitted model
        if isinstance(self.model, xgb.XGBClassifier):
            unfitted_model = XGBClassifierWrapper(**self.model.get_params())
        elif isinstance(self.model, xgb.XGBRegressor):
            unfitted_model = xgb.XGBRegressor(**self.model.get_params())
        else:
            try:
                unfitted_model = clone(self.model)
            except:
                raise ValueError("Unsupported model type")
        
        return unfitted_model

    def evaluate_models(self, model, X_train, y_train, X_val, y_val, X_test, y_test, mode):
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
                res_dict['Accuracy'] = accuracy_score(y, y_pred)
                res_dict['Precision'] = precision_score(y, y_pred)
                res_dict['Recall'] = recall_score(y, y_pred)
                res_dict['F1_score'] = f1_score(y, y_pred)

                try:
                    res_dict['AUC'] = roc_auc_score(y, y_pred)
                except:
                    pass
            
            else:
                res_dict['MSE'] = mean_squared_error(y, y_pred)
                res_dict['MAE'] = mean_absolute_error(y, y_pred)

            tmp_res_df = pd.DataFrame.from_dict(res_dict, orient='index', columns=[name])
            
            if res_df is None:
                res_df = tmp_res_df
            else:
                res_df = pd.concat([res_df, tmp_res_df], axis=1)

        return res_df


    def add_best_feature_set(self):
        """
        Appends the best feature set analysis results to each entry in the results dictionary based on a specified metric.
              
        Returns:
        - results_dict_upd (dict): Updated results dictionary with best feature set analysis appended.
        
        The function iterates over the results dictionary, applies a best feature set selection based on the specified
        main metric, and appends the results back into the dictionary.
        """

        results_dict_upd = self.results_dict.copy()

        for explnr, results in results_dict_upd.items():
            if self.verbose:
                print('\033[1m' + explnr.upper() + '\033[0m')
            results_dict_upd[explnr].append(self.choose_best_feature_set(results[1]))
            if self.verbose:
                print()

        self.results_dict_upd = results_dict_upd


    def choose_best_feature_set(self, model_ev_results, data_type='val'):
        """
        Evaluates and visualizes the best feature set based on a provided metric from model evaluation results.
        
        This function calculates the performance metrics for different numbers of features removed and identifies
        the optimal number of features by finding the highest metric value.

        Returns:
        - num_eliminated_feats (int): The number of features suggested to be removed for optimal performance.
        """
        tdict = {}
        for metric in model_ev_results[0].index:
            metric_list = []
            for i in range(len(model_ev_results)):
                metric_list.append(model_ev_results[i].loc[[metric]][data_type].values[0])
            tdict[metric] = metric_list
        num_eliminated_feats = np.argmax(tdict[self.metric])

        if self.verbose:
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


    def plot_feature_selection_outcomes(self):
        """
        This function generate visualization of performance metrics for selected feature sets evaluated on the test set.

        This function creates two sets of plots: bar charts for feature importance and a table of metrics.
        """

        n_expl = len(self.results_dict_upd)

        # First subplot for bar plots
        fig, axs = plt.subplots(1, n_expl, figsize=(15, 6)) 

        for i, (explnr, results) in enumerate(self.results_dict_upd.items()):
            ax = axs if n_expl == 1 else axs[i]
            results[0][results[2]].plot(kind='barh', ax=ax, legend=False, color='orange', alpha=0.3, edgecolor='#A52A2A')
            ax.set_title(f'{explnr.upper()}\n feature importance\n Best result with \n{results[2]} features removed\n\
{len(results[0][results[2]])} features remain')

        fig.suptitle('Overall Feature Selection Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Second plot for the table
        fig, ax = plt.subplots(figsize=(15, 2)) 
        ax.axis('off')

        # get result without features elimination
        df_expl_results = round(list(self.results_dict_upd.values())[0][1][0]['test'], 4)
        for explnr, results in self.results_dict_upd.items():
            if df_expl_results is None:
                df_expl_results = round(results[1][results[2]]['test'], 4)
            else:
                df_expl_results = pd.concat([df_expl_results, (round(results[1][results[2]]['test'], 4))], axis=1)

        df_expl_results.columns = ['baseline_features_set'] + list(self.results_dict_upd.keys())

        self.df_expl_results = df_expl_results

        # Creating the table
        table = ax.table(cellText=df_expl_results.values, rowLabels=df_expl_results.index, colLabels=df_expl_results.columns, loc='center', cellLoc='center', fontsize=14)

        # Adjusting the table properties manually
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(0.8, 1.4)  # Scale columns width by 1 and row height by 1.5
        ax.set_title(f'Model Metrics for Selected Feature Sets on Test Set')

        plt.show()
