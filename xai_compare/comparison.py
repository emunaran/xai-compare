from abc import ABC, abstractmethod
import pandas as pd
from typing import Union
import numpy as np
import copy

import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score, roc_auc_score, mean_absolute_error, f1_score
from sklearn.base import clone
from sklearn.model_selection import train_test_split
import xgboost as xgb


# Local application imports
from xai_compare.config import MODE, EXPLAINERS
from xai_compare.factory import ExplainerFactory
from xai_compare.explainer_utilities import run_and_collect_explanations_upd


class Comparison(ABC):
    """
    Base class for model comparison that handles various explainer analyses.

    This abstract class provides a framework for comparing explanation methods 
    to assess feature importance and explainer consistency.

    Parameters:
    ----------
    model : model object
        The input machine learning model.
    data : pd.DataFrame
        The feature dataset used for model training and explanation.
    target : Union[pd.DataFrame, pd.Series, np.ndarray]
        The target variables associated with the data.
    custom_explainer : optional
        Custom explainer instances to be added to the default explainers,
        should be made with the framework from explainer.py
    mode : str, default MODE.REGRESSION
        The mode of operation from config.py
    random_state : int, default 42
        Seed used by the random number generator for reproducibility.
    verbose : bool, default True
        Enables verbose output during operations.
    default_explainers : list
        List of default explainers from config.py

    Methods:
    -------
    apply()
        Abstract method to generate a comparison report based on the explainer outputs.
    display()
        Abstract method to plot and display the result from the comparison analysis.
    """

    def __init__(self,
                 model,
                 data: pd.DataFrame,
                 target: Union[pd.DataFrame, pd.Series, np.ndarray],
                 custom_explainer = None,
                 mode: str = MODE.REGRESSION, 
                 random_state=42, 
                 verbose=True,
                 default_explainers=EXPLAINERS):
        self.model = model
        self.data = data
        self.y = target
        self.mode: str = mode
        self.random_state = random_state
        self.verbose = verbose
        self.default_explainers = default_explainers
        self.list_explainers = self.create_list_explainers(custom_explainer)


    def create_list_explainers(self, custom_explainer):
        """
        Creates a list of explainer classes from default and custom explainers.

        Parameters:
        ----------
        custom_explainer : list
            Custom explainer or a list of custom explainer classes.

        Returns:
        -------
        list
            A list of initialized explainer classes.
        """

        list_explainers = [ExplainerFactory().create(explainer_name) for explainer_name in self.default_explainers]

        if custom_explainer:
            list_explainers.extend(custom_explainer)
        
        return list_explainers

    @abstractmethod
    def apply(self):
        pass

    @abstractmethod
    def display(self):
        pass



class Consistency(Comparison):
    """
    A class to evaluate consistency of different explainers on a specified model.

    Attributes:
    - model (Model): The machine learning model to be evaluated.
    - data: data
    - targer: labels.
    - verbose (bool): If True, prints additional information during the function's execution.
    """

    def __init__(self,
                 model,
                 data: pd.DataFrame,
                 target: Union[pd.DataFrame, pd.Series, np.ndarray],
                 custom_explainer = None,
                 mode: str = MODE.REGRESSION, 
                 random_state=42, 
                 verbose=False,
                 n_splits: int = 5, 
                 default_explainers=EXPLAINERS):
        super().__init__(model, data, target, custom_explainer, mode=mode, verbose=verbose, random_state=random_state, \
                         default_explainers=default_explainers) # pass parameters to the parent class
                
        self.n_splits = n_splits
        self.consistency_scores_df = None

    def apply(self):
        if self.consistency_scores_df is  None:
            self.consistency_measurement()
        else:
            pass


    def display(self):
        self.visualize_consistency()

    # def comparison_report(self):
    #     if self.consistency_scores_df is not None:
    #         self.visualize_consistency()
    #     else:
    #         self.consistency_measurement()
    #         self.visualize_consistency()
        

    # def best_result(self):
    #     if self.consistency_scores_df is not None:
    #         return self.consistency_scores_df
    #     else:
    #         self.consistency_measurement()
    #         return self.consistency_scores_df


    def visualize_consistency(self):
        """
        Visualizes the mean and standard deviation of feature impacts for different explainers.
        
        Parameters:
            explainers (list): List of explainer names.
            feature_names (list): List of feature names.
            summary (dict): Dictionary containing mean and standard deviation of feature impacts.
        """

        num_explainers = len(self.list_explainers)
        fig, axes = plt.subplots(1, num_explainers, figsize=(15, 6), sharey=True)

        # Ensure axes is always iterable
        axes = np.atleast_1d(axes)

        # Loop through each explainer and plot the feature impacts
        for ax, explainer in zip(axes, self.list_explainers):
            mean_impact, std_impact = self.summary[explainer.__name__]
            ax.barh(self.data.columns, mean_impact.flatten(), xerr=std_impact.flatten(), align='center', alpha=0.7, ecolor='black', capsize=5)
            ax.set_xlabel('Mean Feature Impact')
            ax.set_title(f'Feature Impact and standard deviation \n {explainer.__name__.upper()}')

        plt.tight_layout()
        plt.show()


    def consistency_measurement(self, stratified_folds=False):
        """
        Measures the consistency of feature explanations across different folds.
        
        Parameters:
            X (DataFrame): Feature matrix.
            y (Series): Target variable.
            model (sklearn estimator): Machine learning model.
            n_splits (int): Number of splits for cross-validation.
            explainers (list): List of explainer names to use.
            verbose (bool): Verbosity flag.
            stratified_folds (bool): Whether to use StratifiedKFold instead of KFold.
        
        Returns:
            DataFrame: DataFrame containing summary statistics of feature impact standard deviations.
        """

        if stratified_folds:
            folds = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        else:
            folds = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        # If no explainers are provided, use all available explainers
        # chosen_explainers = explainers if explainers is not None else available_explainers
        results = {explainer.__name__: [] for explainer in self.list_explainers}

        # Train the model on the full dataset first
        self.model.fit(self.data, self.y)
        
        # Loop through each fold
        for train_index, test_index in tqdm(folds.split(self.data, self.y), total=self.n_splits, desc="Processing folds"):
            X_train, X_test = self.data.iloc[train_index], self.data.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            
            # # Create explainer factory and collect explanations

            for explainer in self.list_explainers:
                explainer_instance = copy.copy(explainer) 
                explainer_instance = explainer_instance(self.model, X_train, y_train, mode=self.mode)
                explainer_values = run_and_collect_explanations_upd(explainer_instance, X_test, verbose=self.verbose)
                results[explainer.__name__].append(explainer_values)

            # # Store the explanation values for each explainer
            # for explainer in chosen_explainers:
            #     explainer_values = explanations[explainer.upper() + " Value"]
            #     results[explainer].append(explainer_values)

        # Calculate mean and standard deviation of feature impacts for each explainer
        summary = {}
        for explainer in self.list_explainers:
            results[explainer.__name__] = np.array(results[explainer.__name__])
            mean_impact = np.mean(results[explainer.__name__], axis=0)
            std_impact = np.std(results[explainer.__name__], axis=0)
            summary[explainer.__name__] = (mean_impact, std_impact)

        self.summary = summary

        # Visualize the consistency of feature impacts
        # visualize_consistency(chosen_explainers, feature_names, summary)

        # Create a DataFrame summarizing the standard deviation statistics for each explainer
        consistency_scores = {}
        for explainer in self.list_explainers:
            key = explainer.__name__
            if key in summary and len(summary[key]) > 1 and len(summary[key][1]) > 0:
                min_std = np.min(summary[key][1])
                max_std = np.max(summary[key][1])
                mean_std = np.mean(summary[key][1])
                median_std = np.median(summary[key][1])
                consistency_scores[key] = {
                    'min_std': min_std,
                    'max_std': max_std,
                    'mean_std': mean_std,
                    'median_std': median_std
                }
            else:
                print(f"No data available for explainer: {key}")

        consistency_scores_df = pd.DataFrame(consistency_scores).T
        
        self.consistency_scores_df = consistency_scores_df



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
                 threshold=0.2,
                 metric=None, 
                 default_explainers=EXPLAINERS):
        super().__init__(model, data, target, custom_explainer, mode=mode, verbose=verbose, random_state=random_state, 
                         default_explainers=default_explainers) # pass parameters to the parent class
                
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

        Parameters:
        ----------
        visualization : bool, optional
            If True, enables the visualization of feature elimination outcomes.

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
        for explainer in tqdm(self.list_explainers, desc="Explainers"):
            results_dict[explainer.__name__] = self.evaluate_explainer(explainer)

        self.results_dict = results_dict


    def evaluate_explainer(self, explainer): 
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
 
        X_train, X_val, X_test = self.X_train, self.X_val, self.X_test

        # clone the model to get unfitted model
        unfitted_model = self.clone_model()

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
            current_importance = run_and_collect_explanations_upd(explainer_instance, X_train, verbose=self.verbose)

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


    def clone_model(self):
        """
        Clones a model based on its type.
        
        Parameters:
        - model: The model to be cloned.

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
        
        Parameters:
        - results_dict (dict): Dictionary containing results for different explainers or methods.
        - mode (str): Operating mode, which determines the main metric ('classification' for accuracy, 'regression' for MSE).
        - visualization (bool, optional): Whether to visualize the results during the process.
        
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