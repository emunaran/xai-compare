# ----------------------------------------------------------------------------------------------------
# Class Explainable Boosting Machine
#
# This class wraps the interpret.glassbox.ExplainableBoostingRegressor and ExplainableBoostingClassifier explainer methods
#
# ------------------------------------------------------------------------------------------------------
import interpret.glassbox
import pandas as pd
import numpy as np
from typing import Union

# Local application imports
from xai_compare.abstract.explainer import Explainer
from xai_compare.config import MODE


class EBM(Explainer):
    """
    A class that encapsulates the Explainable Boosting Machine (EBM) from the interpret community's glassbox models.

    Attributes:
        model: An input machine learning model.
        X_train (pd.DataFrame): Training data features.
        y_train (Union[pd.DataFrame, pd.Series, np.ndarray]): Training data labels.
        y_pred (Union[pd.DataFrame, pd.Series, np.ndarray, None], optional): Predicted labels (used in some contexts).
        mode (str): Indicates whether the explainer is used for 'regression' or 'classification'.
    """
    def __init__(self, model, X_train: pd.DataFrame, y_train: Union[pd.DataFrame, pd.Series, np.ndarray], 
                 y_pred: Union[pd.DataFrame, pd.Series, np.ndarray, None] = None, mode: str = MODE.REGRESSION):
        super().__init__(model, X_train, y_train, y_pred, mode)
        self._create_explainer()

    def _create_explainer(self) -> None:
        """
        Creates an Explainable Boosting Machine (EBM) model based on the mode and fits it with the training data.
        """

        if self.mode == MODE.REGRESSION:
            self.explainer = interpret.glassbox.ExplainableBoostingRegressor()
        else:
            self.explainer = interpret.glassbox.ExplainableBoostingClassifier()
        
        self.explainer.fit(self.X_train, self.y_train)
    
    def predict(self, X_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts the target for the provided feature data using the EBM model.

        Parameters:
            X_data (pd.DataFrame): The input features for which predictions are to be made.

        Returns:
            pd.DataFrame: The predictions made by the EBM model.
        """
        return self.explainer.predict(X_data)
    
    def predict_proba(self, X_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts class probabilities for the provided feature data using the EBM model. Only applicable in classification mode.

        Parameters:
            X_data (pd.DataFrame): The input features for which class probabilities are to be predicted.

        Returns:
            pd.DataFrame: The class probabilities predicted by the EBM model.

        Raises:
            NotImplementedError: If the method is called in regression mode.
        """

        if self.mode != MODE.REGRESSION:
            return self.explainer.predict_proba(X_data)
        else:
            raise NotImplementedError('predict_proba is not available for the regression mode')

    def explain_global(self, X_data: pd.DataFrame) -> pd.DataFrame:
        """
        Provides a global explanation of the model where the importance of each feature is summarized across all instances.

        Parameters:
            X_data (pd.DataFrame): The dataset for which global explanations are generated.

        Returns:
            pd.DataFrame: A DataFrame containing global importance scores for each feature.
        """
        return pd.DataFrame(self.explainer.explain_global().data()['scores'][:len(X_data.columns)], 
                            index=X_data.columns, columns=['EBM Value'])

    def explain_local(self, X_data: pd.DataFrame) -> pd.DataFrame:
        """
        Provides local explanations for each instance in the dataset, explaining the contribution of each feature to the individual prediction.

        Parameters:
            X_data (pd.DataFrame): The dataset for which local explanations are generated.

        Returns:
            pd.DataFrame: A DataFrame where each row represents an instance with feature contributions for that specific prediction.
        """
        return self.explainer.explain_local(X_data)


