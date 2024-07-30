# ----------------------------------------------------------------------------------------------------
# Class Permutations
#
# This class wraps the permutation explainer
#
# ------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from typing import Union
from sklearn.metrics import accuracy_score, mean_squared_error

# Local application imports
from xai_compare.abstract.explainer import Explainer
from xai_compare.config import MODE


class Permutations(Explainer):
    """
    A class that encapsulates the Permutation Feature Importance method for explaining model predictions.
    Further details can be found in the literature, such as in the book "Interpretable Machine Learning"
    by Christoph Molnar, which discusses various interpretation methods including permutation importance.
    (https://christophm.github.io/interpretable-ml-book/feature-importance.html)

    Attributes:
        model:
            An input machine learning model.

        X_train (pd.DataFrame):
            Training data features.

        y_train (Union[pd.DataFrame, pd.Series, np.ndarray]):
            Training data labels.

        mode (str):
            Indicates whether the explainer is used for 'regression' or 'classification'.

        num_permutations (int):
            Number of permutations to perform for each feature.

        random_state (Union[int, None]):
            Seed for the random number generator.
    """

    __name__ = "permutations"

    def __init__(self, 
                 model, 
                 X_train: pd.DataFrame, 
                 y_train: Union[pd.DataFrame, pd.Series, np.ndarray],
                 mode: str = MODE.REGRESSION, 
                 num_permutations: int = 5, 
                 random_state: Union[int, None] = None):
        super().__init__(model, X_train, y_train, mode=mode) # pass parameters to the parent class
        
        self.mode = MODE.REGRESSION
        self.num_permutations = num_permutations
        self.random_state = random_state

    def explain_global(self, X_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates permutation feature importance for a given model.

        Attributes:
            X_data (pd.DataFrame):
                The feature matrix for which permutation importance is calculated.

        Returns:
            pd.DataFrame:
                A DataFrame containing the feature names and their mean importance.
        """
        np.random.seed(self.random_state)

        if self.mode == MODE.CLASSIFICATION:
            scorer = accuracy_score
        elif self.mode == MODE.REGRESSION:
            scorer = mean_squared_error

        # Initialize a dictionary to store importance scores for each feature
        feature_importances = {feature: [] for feature in X_data.columns}
        
        # Loop over each feature to permute and evaluate its importance
        for feature in X_data.columns:
            for _ in range(self.num_permutations):
                # Permute the current feature
                X_data_permuted = X_data.copy()
                X_data_permuted[feature] = np.random.permutation(X_data[feature])
                # Evaluate the model with the permuted feature and calculate the importance score
                importance_score = scorer(self.model.predict(X_data), self.model.predict(X_data_permuted))
                feature_importances[feature].append(importance_score)

        # Average the importance scores over the number of repeats
        averaged_importances = {feature: np.mean(importances) for feature, importances in feature_importances.items()}
        
        # Normalize the importance scores to sum to 0.5 (for comparability with other explainers)
        total_importance = sum(averaged_importances.values())
        normalized_importances = {feature: (importance / total_importance) * 0.5 for feature, importance in averaged_importances.items()}

        # Convert to DataFrame 
        feature_importances_df = pd.DataFrame.from_dict(normalized_importances, orient='index', columns=['importance'])
        
        return feature_importances_df

    def explain_local(self, X_data: pd.DataFrame) -> pd.DataFrame:
        """
        This function is not applicable for permutation feature importance as it is a global method.

        Attributes:
            Not applicable.

        Returns:
            None:
                This function does not return any value.
        """
        raise NotImplementedError("The explain_local method is not applicable for permutation feature importance.")