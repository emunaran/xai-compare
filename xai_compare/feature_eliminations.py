import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score, roc_auc_score, mean_absolute_error, f1_score
from sklearn.base import clone
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Union

# Local application imports
from xai_compare.explainer_factory import ExplainerFactory
from xai_compare.explainer_utilities import run_and_collect_explanations_upd
from xai_compare.config import MODE
from xai_compare.comparison import Comparison


class FeatureElimination(Comparison):
    """
    A class to evaluate different feature elimination strategies provided by the list of explainers on a specified model.

    Attributes:
    - model (Model): The machine learning model to be evaluated.
    - data: data
    - targer: labels.
    - mode (str): The mode of operation for the explainers ('classification', 'regression').
    - threshold (float): The threshold for feature importance below which features are considered for elimination.
    - random_state (int): A seed value to ensure reproducibility.
    - verbose (bool): If True, prints additional information during the function's execution.
    """

    def __init__(self,
                 model,
                 data: pd.DataFrame,
                 target: Union[pd.DataFrame, pd.Series, np.ndarray],
                 custom_explainer = None,
                 mode: str = MODE.REGRESSION, 
                 random_state=42, 
                 verbose=True,
                 threshold=0.2):
        super().__init__(model, data, target, custom_explainer, mode=mode, verbose=verbose, random_state=random_state) # pass parameters to the parent class
                
        self.threshold = threshold
        self.results = {}
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.train_test_val()

    def train_test_val(self):
        X_train, X_tmp, y_train, y_tmp = train_test_split(self.data, self.y, test_size=0.5, random_state=self.random_state)  
        X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.4, random_state=self.random_state)  
        return X_train, y_train, X_val, y_val, X_test, y_test


    def comparison_report(self):
        pass


    def best_result(self):
        pass


    def get_feature_elimination_results(self):
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

        for explainer in self.list_explainers:
            results_dict[explainer.__name__] = self.evaluate_explainer(explainer)

        self.results_dict = results_dict
        
        return results_dict
    

    def evaluate_explainer(self, explainer): #metric='mse', 
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
        current_model = self.model

        X_train, X_val, X_test = self.X_train, self.X_val, self.X_test

        # clone the model to get unfitted model
        unfitted_model = clone(self.model)

        if self.mode == MODE.CLASSIFICATION:
            self.metric = 'accuracy'
        else:
            self.metric = 'mse'

        # Get the list of column names
        columns = self.X_train.columns.tolist()
        n = len(columns)
        remaining_features = n * self.threshold

        # Initialize variables to store resuts
        res_list = [] # list of feature importance
        list_el_feats = [] # list of least important features

        res_model_eval = []

        # Loop until the number of features is reduced to the desired threshold
        while len(columns) > remaining_features:
            
            current_model.fit(X_train, self.y_train)

            # evaluate the model
            current_model_results = self.evaluate_models(current_model, X_train, self.y_train, X_val, \
                                                    self.y_val, X_test, self.y_test, self.mode)
            res_model_eval.append(current_model_results)
        
            # Get explainer values
            # explainer_factory = ExplainerFactory(current_model, X_train=X_train, y_train=y_train)
            current_importance = run_and_collect_explanations_upd(explainer, X_train, verbose=self.verbose)

            # Find the least important feature
            # - apply absolute value on the global explanation results to get the feature importance for each XAI method.
            feature_importances = abs(current_importance)

            # - sort the list to have the order of the least to the most important features.
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


    def add_best_feature_set(self, visualization=True):
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

        if self.mode == MODE.CLASSIFICATION:
            main_metric = 'Accuracy'
        else:
            main_metric = 'MSE'

        results_dict_upd = self.results_dict.copy()

        for explnr,results in results_dict_upd.items():
            print('\033[1m' + explnr.upper() + '\033[0m')
            results_dict_upd[explnr].append(choose_best_feature_set(results[1], main_metric, visualization=visualization))
            print()

        self.results_dict_upd = results_dict_upd

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


    def plot_feature_selection_outcomes(self):
        """
        This function generate visualization of performance metrics for selected feature sets evaluated on the test set.

        Parameters:
        - results_dict_upd (dict): A dictionary where each key is an explainer name, and the value is
        a list containing a DataFrame of feature importances and model performance metrics for each feature set.

        This function creates two sets of plots: bar charts for feature importance and a table of metrics.
        """

        n_expl = len(self.results_dict_upd)

        # First subplot for bar plots
        fig, axs = plt.subplots(1, n_expl, figsize=(15, 6)) 

        for i, (explnr, results) in enumerate(self.results_dict_upd.items()):
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
        df_expl_results = round(list(self.results_dict_upd.values())[0][1][0]['test'], 4)
        for explnr, results in self.results_dict_upd.items():
            if df_expl_results is None:
                df_expl_results = round(results[1][results[2]]['test'], 4)
            else:
                df_expl_results = pd.concat([df_expl_results,(round(results[1][results[2]]['test'], 4))], axis=1)

        df_expl_results.columns = ['baseline_features_set'] + list(self.results_dict_upd.keys())

        pd.plotting.table(ax, df_expl_results, loc='upper center', colWidths=[0.15] * len(df_expl_results.columns))
        ax.set_title(f'Model Metrics for Selected Feature Sets on Test Set')

        plt.show()


    # def evaluate_model(self, model, X_train, y_train, X_val, y_val, X_test, y_test):
    #     """
    #     Evaluates a model's performance on training, validation, and testing datasets.

    #     Parameters:
    #     - model: The model to be evaluated.
    #     - X_train, y_train, X_val, y_val, X_test, y_test: Data for evaluation.

    #     Returns:
    #     - DataFrame: A DataFrame containing performance metrics.
    #     """
    #     y_pred_train = model.predict(X_train)
    #     y_pred_val = model.predict(X_val)
    #     y_pred_test = model.predict(X_test)
    #     if self.mode == 'classification':
    #         return {
    #             'train_accuracy': accuracy_score(y_train, y_pred_train),
    #             'val_accuracy': accuracy_score(y_val, y_pred_val),
    #             'test_accuracy': accuracy_score(y_test, y_pred_test)
    #         }
    #     else:
    #         return {
    #             'train_mse': mean_squared_error(y_train, y_pred_train),
    #             'val_mse': mean_squared_error(y_val, y_pred_val),
    #             'test_mse': mean_squared_error(y_test, y_pred_test)
    #         }

    # def calculate_feature_importance(self, model, X, y, explainer):
    #     """
    #     Calculates feature importance using the specified explainer method.

    #     Parameters:
    #     - model: The model for which to calculate importance.
    #     - X, y: Data used for importance calculation.
    #     - explainer: The explainer method to use.

    #     Returns:
    #     - Series: A Series of feature importances.
    #     """
    #     # This function should be implemented based on the specific type of explainer being used.
    #     # Here, we simulate a dummy feature importance calculation
    #     importances = np.random.rand(len(X.columns))
    #     return pd.Series(importances, index=X.columns)


    

# def get_feature_elimination_results(list_explainers, model, X_train, y_train, X_val, y_val, \
#                                     X_test, y_test, mode, threshold=0.2, random_state=None, verbose=True):
#     """
#     Evaluates different feature elimination strategies provided by the list of explainers on a specified model.
    
#     Each explainer is used to assess the importance of features, and based on that, evaluate the model's performance 
#     with progressively eliminated features.

#     Parameters:
#     - list_explainers (list): A list of explainer instances or identifiers used to evaluate feature importance.
#     - model: The machine learning model to be evaluated.
#     - X_train, y_train, X_val, y_val, X_test, y_test: Training, validation, and testing datasets.
#     - mode (str): The mode of operation for the explainers (e.g., 'classification', 'regression').
#     - threshold (float, optional): The threshold for feature importance below which features are considered for elimination. Defaults to 0.2.
#     - random_state (int, optional): A seed value to ensure reproducibility. Defaults to None.
#     - verbose (bool, optional): If True, prints additional information during the function's execution. Defaults to True.

#     Returns:
#     - results_dict (dict): A dictionary containing the results from each explainer.
#     """


#     results_dict = {}

#     for explainer in list_explainers:
#         results_dict[explainer] = evaluate_explainer(model, X_train, y_train, X_val, y_val, X_test, y_test,\
#                                                                 explainer, mode, threshold=threshold, \
#                                                                     random_state=random_state, verbose=verbose)
    
#     return results_dict

# def evaluate_explainer(model, X_train, y_train, X_val, y_val, X_test, y_test, explainer, mode, threshold=0.2, random_state=None, verbose=True,): #metric='mse', 
#     """
#     Evaluates the performance of a machine learning model with progressively fewer features based on the importance
#     determined by various explainer methods.

#     This function iteratively eliminates the least important features as determined by the specified explainer
#     until the number of features is reduced to the desired threshold.

#     Parameters:
#     - model: Trained model to be evaluated.
#     - X_train, y_train: Training data and labels.
#     - X_val, y_val: Validation data and labels.
#     - X_test, y_test: Test data and labels.
#     - explainer (str): The type of explainer to use for feature importance evaluation.
#     - mode (str): The mode of the operation, typically 'classification' or 'regression'.
#     - threshold (float): Proportion of features to retain based on their importance.
#     - random_state (int, optional): Seed used by random number generators for reproducibility.
#     - verbose (bool, optional): If True, prints detailed progress information.

#     Returns:
#     - A list containing a list of DataFrames with feature importances and model evaluation results.
#     """
#     current_model = model

#     # clone the model to get unfitted model
#     unfitted_model = clone(model)

#     if mode == MODE.CLASSIFICATION:
#         metric = 'accuracy'
#     else:
#         metric = 'mse'

#     # Get the list of column names
#     columns = X_train.columns.tolist()
#     n = len(columns)
#     remaining_features = n * threshold

#     # Initialize variables to store resuts
#     res_list = [] # list of feature importance
#     list_el_feats = [] # list of least important features

#     res_model_eval = []

#     # Loop until the number of features is reduced to the desired threshold
#     while len(columns) > remaining_features:

#         # evaluate the model
#         current_model_results = evaluate_models(current_model, X_train, y_train, X_val, \
#                                                 y_val, X_test, y_test, mode)
#         res_model_eval.append(current_model_results)
    
#         # Get explainer values
#         explainer_factory = ExplainerFactory(current_model, X_train=X_train, y_train=y_train)
#         current_importance = run_and_collect_explanations(explainer_factory, X_train, verbose=verbose, explainers=explainer)

#         # Find the least important feature
#         # - apply absolute value on the global explanation results to get the feature importance for each XAI method.
#         feature_importances = abs(current_importance)

#         # - sort the list to have the order of the least to the most important features.
#         sorted_feature_importances = feature_importances.sort_values(by=current_importance.columns[0], ascending=True)
#         least_important_feature = sorted_feature_importances.index[0]

#         # Log progress
#         if verbose:
#             print(f'Iteration: {len(columns) - len(X_train.columns) + 1}, Removed: {least_important_feature}')


#         # results = pd.concat([current_importance, feature_importances_df], axis=1)
#         results = sorted_feature_importances

#         res_list.append(results) # add feature importance into the list

#         list_el_feats.append(least_important_feature)

#         # Drop the least important feature
#         X_train = X_train.drop(columns=[least_important_feature])
#         X_val = X_val.drop(columns=[least_important_feature])
#         X_test = X_test.drop(columns=[least_important_feature])
#         columns.remove(least_important_feature)

#         # Retrain the model with the reduced feature set
#         current_model = unfitted_model
#         current_model.fit(X_train, y_train)

#     # return a list of DataFrames with model evaluation results
#     return [res_list, res_model_eval]


# def evaluate_models(model, X_train, y_train, X_val, y_val, X_test, y_test, mode):
#     """
#     Evaluates a model's performance metrics on training, validation, and test datasets.
    
#     Parameters:
#     - model: The model to evaluate.
#     - X_train, y_train: Training data and labels.
#     - X_val, y_val: Validation data and labels.
#     - X_test, y_test: Test data and labels.
#     - mode (str): Operation mode, 'classification' or 'regression'.

#     Returns:
#     - res_df (DataFrame): A DataFrame containing performance metrics for each dataset.

#     The function calculates accuracy, precision, recall, and F1 scores for classification mode,
#     and mean squared error and mean absolute error for regression mode.
#     """

#     res_dict, res_df = {}, None
#     y_pred_train = model.predict(X_train)
#     y_pred_val = model.predict(X_val)
#     y_pred_test = model.predict(X_test)
#     train_tpl = (y_train, y_pred_train, 'train')
#     val_tpl = (y_val, y_pred_val, 'val')
#     test_tpl = (y_test, y_pred_test, 'test')

#     for y, y_pred, name in [train_tpl, val_tpl, test_tpl]:

#         if mode == MODE.CLASSIFICATION:
#             res_dict['Accuracy'] = accuracy_score(y, y_pred)
#             res_dict['Precision'] = precision_score(y, y_pred)
#             res_dict['Recall'] = recall_score(y, y_pred)
#             res_dict['F1_score'] = f1_score(y, y_pred)

#             try:
#                 res_dict['AUC'] = roc_auc_score(y, y_pred)
#             except:
#                 pass
        
#         else:
#             res_dict['MSE'] = mean_squared_error(y, y_pred)
#             res_dict['MAE'] = mean_absolute_error(y, y_pred)

#         tmp_res_df = pd.DataFrame.from_dict(res_dict, orient='index', columns=[name])
        
#         if res_df is None:
#             res_df = tmp_res_df
#         else:
#             res_df = pd.concat([res_df, tmp_res_df], axis=1)

#     return res_df


# def add_best_feature_set(results_dict, mode, visualization=True):
#     """
#     Appends the best feature set analysis results to each entry in the results dictionary based on a specified metric.
    
#     Parameters:
#     - results_dict (dict): Dictionary containing results for different explainers or methods.
#     - mode (str): Operating mode, which determines the main metric ('classification' for accuracy, 'regression' for MSE).
#     - visualization (bool, optional): Whether to visualize the results during the process.
    
#     Returns:
#     - results_dict_upd (dict): Updated results dictionary with best feature set analysis appended.
    
#     The function iterates over the results dictionary, applies a best feature set selection based on the specified
#     main metric, and appends the results back into the dictionary.
#     """

#     if mode == MODE.CLASSIFICATION:
#         main_metric = 'Accuracy'
#     else:
#         main_metric = 'MSE'

#     results_dict_upd = results_dict.copy()

#     for explnr,results in results_dict_upd.items():
#         print('\033[1m' + explnr.upper() + '\033[0m')
#         results_dict_upd[explnr].append(choose_best_feature_set(results[1], main_metric, visualization=visualization))
#         print()

#     return results_dict_upd


# def choose_best_feature_set(model_ev_results, main_metric, data_type = 'val', visualization=True):
#     """
#     Evaluates and visualizes the best feature set based on a provided metric from model evaluation results.
    
#     This function calculates the performance metrics for different numbers of features removed and identifies
#     the optimal number of features by finding the highest metric value.

#     Parameters:
#     - model_ev_results (list): A list of DataFrame objects containing model evaluation metrics.
#     - main_metric (str): The metric name to evaluate for the best feature set (e.g., 'accuracy', 'mse').
#     - data_type (str, optional): Specifies the type of data ('train', 'val', or 'test') on which metrics are based.
#     - visualization (bool, optional): If True, generates a plot to visualize the metrics across different feature sets.

#     Returns:
#     - num_eliminated_feats (int): The number of features suggested to be removed for optimal performance.
#     """

#     tdict = {}
#     for metric in model_ev_results[0].index:
#         metric_list = []
#         for i in range(len(model_ev_results)):
#             metric_list.append(model_ev_results[i].loc[[metric]][data_type].values[0])
#         tdict[metric] = metric_list
#     num_eliminated_feats = np.argmax(tdict[main_metric])

#     if visualization:
#         fig, ax = plt.subplots(figsize=(8, 5))

#         tdf = pd.DataFrame(tdict)

#         tdf.plot(ax=ax)

#         plt.axvline(x=num_eliminated_feats, color='r', linestyle='--', label='Best feature set')

#         # Set y-axis limits
#         ax.set_ylim(0, 1)  # Set the limits of the y-axis to be from 0 to 1

#         # Set axis labels
#         ax.set_xlabel('Number of Features Eliminated')  
#         ax.set_ylabel('Evaluation Metric Value')      

#         # Set the plot title
#         ax.set_title('Feature Elimination Analysis') 

#         plt.show()

#         print(f'{num_eliminated_feats} features are suggested to be removed')
#         print(model_ev_results[num_eliminated_feats])

#     return num_eliminated_feats  


# def plot_feature_selection_outcomes(results_dict_upd):
#     """
#     This function generate visualization of performance metrics for selected feature sets evaluated on the test set.

#     Parameters:
#     - results_dict_upd (dict): A dictionary where each key is an explainer name, and the value is
#       a list containing a DataFrame of feature importances and model performance metrics for each feature set.

#     This function creates two sets of plots: bar charts for feature importance and a table of metrics.
#     """

#     n_expl = len(results_dict_upd)

#     # First subplot for bar plots
#     fig, axs = plt.subplots(1, n_expl, figsize=(15, 6)) 

#     for i, (explnr, results) in enumerate(results_dict_upd.items()):
#         ax = axs if n_expl == 1 else axs[i]
#         results[0][results[2]].plot(kind='barh', ax=ax, legend=False)
#         ax.set_title(f'{explnr.upper()}\n feature importance\n Best result with \n{results[2]} features removed\n')

#     fig.suptitle('Overall Feature Selection Results', fontsize=16, fontweight='bold')
#     plt.tight_layout()
#     plt.show()

#     # Second plot for the table
#     fig, ax = plt.subplots(figsize=(15, 2)) 
#     ax.axis('off')

#     # get result without features elimination
#     df_expl_results = round(list(results_dict_upd.values())[0][1][0]['test'], 4)
#     for explnr, results in results_dict_upd.items():
#         if df_expl_results is None:
#             df_expl_results = round(results[1][results[2]]['test'], 4)
#         else:
#             df_expl_results = pd.concat([df_expl_results,(round(results[1][results[2]]['test'], 4))], axis=1)

#     df_expl_results.columns = ['baseline_features_set'] + list(results_dict_upd.keys())

#     pd.plotting.table(ax, df_expl_results, loc='upper center', colWidths=[0.15] * len(df_expl_results.columns))
#     ax.set_title(f'Model Metrics for Selected Feature Sets on Test Set')

#     plt.show()