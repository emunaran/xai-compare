import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Union
import copy

# Local application imports
from xai_compare.explainer_factory import ExplainerFactory
from xai_compare.explainer_utilities import run_and_collect_explanations_upd
from xai_compare.comparison import Comparison
from xai_compare.config import MODE, EXPLAINERS


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

    def comparison_report(self):
        if self.consistency_scores_df is not None:
            self.visualize_consistency()
        else:
            self.consistency_measurement()
            self.visualize_consistency()
        

    def best_result(self):
        if self.consistency_scores_df is not None:
            return self.consistency_scores_df
        else:
            self.consistency_measurement()
            return self.consistency_scores_df


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

        available_explainers = ["shap", "lime"] # Easily extendable for additional explainers
        feature_names = self.data.columns

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
            # factory = ExplainerFactory(self.model, X_train, X_test, y_train, y_test)
            # explanations = run_and_collect_explanations(factory, X_train, explainers=chosen_explainers, verbose=verbose)

            for explainer in self.list_explainers:
                explainer_instance = copy.copy(explainer) 
                explainer_instance = explainer_instance(self.model, X_train, y_train)
                explainer_values = run_and_collect_explanations_upd(explainer_instance, X_train, verbose=self.verbose)
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


        # consistency_scores = {explainer.__name__: {'min_std': np.min(summary[explainer.__name__][1]), 
        #                             'max_std': np.max(summary[explainer.__name__][1]),
        #                             'mean_std': np.mean(summary[explainer.__name__][1]), 
        #                             'median_std': np.median(summary[explainer.__name__][1])} for explainer in self.list_explainers}
        consistency_scores_df = pd.DataFrame(consistency_scores).T
        
        self.consistency_scores_df = consistency_scores_df


# def visualize_consistency(explainers, feature_names, summary):
#     """
#     Visualizes the mean and standard deviation of feature impacts for different explainers.
    
#     Parameters:
#         explainers (list): List of explainer names.
#         feature_names (list): List of feature names.
#         summary (dict): Dictionary containing mean and standard deviation of feature impacts.
#     """

#     num_explainers = len(explainers)
#     fig, axes = plt.subplots(1, num_explainers, figsize=(15, 6), sharey=True)

#     # Loop through each explainer and plot the feature impacts
#     for ax, explainer in zip(axes, explainers):
#         mean_impact, std_impact = summary[explainer]
#         ax.barh(feature_names, mean_impact, xerr=std_impact, align='center', alpha=0.7, ecolor='black', capsize=5)
#         ax.set_xlabel('Mean Feature Impact')
#         ax.set_title(f'Feature Impact and standard deviation - {explainer.upper()}')

#     plt.tight_layout()
#     plt.show()


# def consistency_measurement(X, y, model, n_splits=5, explainers=None, verbose=False, stratified_folds=False):
#     """
#     Measures the consistency of feature explanations across different folds.
    
#     Parameters:
#         X (DataFrame): Feature matrix.
#         y (Series): Target variable.
#         model (sklearn estimator): Machine learning model.
#         n_splits (int): Number of splits for cross-validation.
#         explainers (list): List of explainer names to use.
#         verbose (bool): Verbosity flag.
#         stratified_folds (bool): Whether to use StratifiedKFold instead of KFold.
    
#     Returns:
#         DataFrame: DataFrame containing summary statistics of feature impact standard deviations.
#     """

#     if stratified_folds:
#         folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#     else:
#         folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)

#     available_explainers = ["shap", "lime"] # Easily extendable for additional explainers
#     feature_names = X.columns

#     # If no explainers are provided, use all available explainers
#     chosen_explainers = explainers if explainers is not None else available_explainers
#     results = {explainer: [] for explainer in chosen_explainers}

#     # Train the model on the full dataset first
#     model.fit(X, y)
    
#     # Loop through each fold
#     for train_index, test_index in tqdm(folds.split(X, y), total=n_splits, desc="Processing folds"):
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
#         # Create explainer factory and collect explanations
#         factory = ExplainerFactory(model, X_train, X_test, y_train, y_test)
#         explanations = run_and_collect_explanations(factory, X_train, explainers=chosen_explainers, verbose=verbose)

#         # Store the explanation values for each explainer
#         for explainer in chosen_explainers:
#             explainer_values = explanations[explainer.upper() + " Value"]
#             results[explainer].append(explainer_values)

#     # Calculate mean and standard deviation of feature impacts for each explainer
#     summary = {}
#     for explainer in explainers:
#         results[explainer] = np.array(results[explainer])
#         mean_impact = np.mean(results[explainer], axis=0)
#         std_impact = np.std(results[explainer], axis=0)
#         summary[explainer] = (mean_impact, std_impact)

#     # Visualize the consistency of feature impacts
#     visualize_consistency(chosen_explainers, feature_names, summary)

#     # Create a DataFrame summarizing the standard deviation statistics for each explainer
#     consistency_scores = {explainer: {'min_std': np.min(summary[explainer][1]), 
#                                 'max_std': np.max(summary[explainer][1]),
#                                 'mean_std': np.mean(summary[explainer][1]), 
#                                 'median_std': np.median(summary[explainer][1])} for explainer in explainers}
#     consistency_scores_df = pd.DataFrame(consistency_scores).T
    
#     return consistency_scores_df