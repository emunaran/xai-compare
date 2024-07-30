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
from typing import Union, List, Type
from sklearn.model_selection import KFold, StratifiedKFold

# Local application imports
from xai_compare.config import MODE, EXPLAINERS
from xai_compare.abstract.explainer import Explainer
from xai_compare.explainer_utilities import run_and_collect_explanations
from xai_compare.abstract.comparison import Comparison


class Consistency(Comparison):
    """
    A class to evaluate the consistency of different explainers on a specified model.

    Attributes:
        model (Model):
            The machine learning model to be evaluated.

        data (pd.DataFrame):
            The feature dataset used for model training and explanation.

        target (Union[pd.DataFrame, pd.Series, np.ndarray]):
            The target variables associated with the data.

        mode (str):
            The mode of operation, either 'REGRESSION' or 'CLASSIFICATION'.

        random_state (int):
            Seed used by the random number generator.

        verbose (bool):
            If True, prints additional information during the function's execution.

        n_splits (int):
            The number of splits for cross-validation.

        use_stratified_folds (bool):
            If True, uses stratified folds for cross-validation.

        shuffle (bool):
            If True, shuffles the data before splitting into batches.

        default_explainers (List[str]):
            List of default explainers to be used.

        custom_explainer (Union[Type[Explainer], List[Type[Explainer]], None]):
            Custom explainer(s) provided by the user.

        consistency_scores_df (pd.DataFrame, optional):
            DataFrame containing the summary statistics of feature impact standard deviations.

        scores (Any, optional):
            Placeholder for the scores obtained during evaluation.

        summary (Any, optional):
            Placeholder for the summary of results.

        results (Any, optional):
            Placeholder for the detailed results.

        comparison_plot (Any, optional):
            Placeholder for the comparison plot.
    """

    def __init__(self,
                 model,
                 data: pd.DataFrame,
                 target: Union[pd.DataFrame, pd.Series, np.ndarray],
                 mode: str = MODE.REGRESSION, 
                 random_state: int = 42, 
                 verbose: bool = False,
                 n_splits: int = 5,
                 use_stratified_folds: bool = False,
                 shuffle: bool = False,
                 default_explainers: List[str] = EXPLAINERS,
                 custom_explainer: Union[Type[Explainer], List[Type[Explainer]], None] = None):
        
        super().__init__(model, data, target, mode=mode, random_state=random_state, verbose=verbose, 
                         default_explainers=default_explainers, custom_explainer=custom_explainer)

        if not isinstance(n_splits, int) or n_splits <= 1:
            raise ValueError("n_splits should be an integer greater than 1.")
    
        self.n_splits = n_splits
        self.use_stratified_folds = use_stratified_folds
        self.shuffle = shuffle
        self.consistency_scores_df = None
        self.scores = None
        self.summary = None
        self.results = None
        self.comparison_plot = None

    def apply(self):
        """
        Applies the consistency measurement if it has not been done yet.
        """
        if self.consistency_scores_df is None:
            self.consistency_measurement(self.use_stratified_folds)
        else:
            pass

    def display(self):
        """
        Displays the consistency measurement results.
        """
        self.visualize_consistency()
        print(self.scores)  

    def visualize_consistency(self):
        """
        Visualizes the mean and standard deviation of feature impacts for different explainers.
        """
        if not hasattr(self, 'summary'):
            raise AttributeError("The consistency_measurement method should be called before visualization.")
          
        num_explainers = len(self.list_explainers)
        fig, axes = plt.subplots(1, num_explainers, figsize=(15, 6), sharey=True)

        # Ensure axes is always iterable
        axes = np.atleast_1d(axes)

        # Loop through each explainer and plot the feature impacts
        for ax, explainer in zip(axes, self.list_explainers):
            mean_impact, std_impact = self.summary[explainer.__name__]
            ax.barh(self.data.columns, mean_impact.flatten(), xerr=std_impact.flatten(), align='center', alpha=0.3, color='orange', edgecolor='#A52A2A', ecolor='black', capsize=5)
            ax.set_xlabel('Mean Feature Impact')
            ax.set_title(f'Feature Impact and standard deviation \n {explainer.__name__.upper()}')

        fig.tight_layout()

        self.comparison_plot = fig
        plt.show()

    def consistency_measurement(self, use_stratified_folds):
        """
        Measures the consistency of feature explanations across different folds.
        
        Parameters:
            use_stratified_folds (bool): Whether to use StratifiedKFold instead of KFold.

        Returns:
            DataFrame: DataFrame containing summary statistics of feature impact standard deviations.
        """
        
        if use_stratified_folds and self.mode != MODE.CLASSIFICATION:
            raise ValueError("StratifiedKFold is only applicable for classification mode.")

        self.random_state = self.random_state if self.shuffle else None

        if use_stratified_folds:
            folds = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        else:
            folds = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)

        results = {explainer.__name__: [] for explainer in self.list_explainers}

        # Train the model on the full dataset first
        self.model.fit(self.data, self.y)
        
        # Loop through each fold
        for train_index, test_index in tqdm(folds.split(self.data, self.y), total=self.n_splits, desc="Processing folds"):
            X_train, X_test = self.data.iloc[train_index], self.data.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            
            # Create explainer factory and collect explanations

            for explainer in self.list_explainers:
                explainer_instance = copy.copy(explainer)
                explainer_instance = explainer_instance(self.model, X_train, y_train, mode=self.mode)
                explainer_values = run_and_collect_explanations(explainer_instance, X_test, verbose=self.verbose)
                results[explainer.__name__].append(explainer_values)
        self.results = results

        # Calculate mean and standard deviation of feature impacts for each explainer
        summary = {}
        for explainer in self.list_explainers:
            results[explainer.__name__] = np.array(results[explainer.__name__])
            mean_impact = np.mean(results[explainer.__name__], axis=0)
            std_impact = np.std(results[explainer.__name__], axis=0)
            summary[explainer.__name__] = (mean_impact, std_impact)

        self.summary = summary

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
        self.scores = consistency_scores_df

