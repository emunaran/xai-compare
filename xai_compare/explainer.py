# ----------------------------------------------------------------------------------------------------
# Class Explainer
# This is the abstract class with explain_global adn explain_local as the abstract methods
# SHAP, LIME and XGBOOST inherit from this class
# ------------------------------------------------------------------------------------------------------
from abc import ABC, abstractmethod
import pandas as pd
from typing import Union
import numpy as np

# Local application imports
from xai_compare.config import MODE


class Explainer(ABC):
    """
    An abstract base class for creating explainers that can interpret the predictions made by machine learning models.

    Attributes:
        model: A machine learning model which predictions are to be interpreted.
        X_train (pd.DataFrame): Training data used to fit the model.
        y_train (Union[pd.DataFrame, pd.Series, np.ndarray]): Training labels or targets.
        y_pred (Union[pd.DataFrame, pd.Series, np.ndarray, None], optional): Predicted values. Defaults to None.
        mode (str): The mode of the explainer which could be 'regression' or 'classification' from config.py.

    Methods:
        explain_global(x_data: pd.DataFrame) -> pd.DataFrame: Abstract method to compute global explanations.
        explain_local(x_data: pd.DataFrame) -> pd.DataFrame: Abstract method to compute local explanations.
    """

    def __init__(self,
                 model,
                 X_train: pd.DataFrame,
                 y_train: Union[pd.DataFrame, pd.Series, np.ndarray],
                 y_pred: Union[pd.DataFrame, pd.Series, np.ndarray, None] = None,
                 mode: str = MODE.REGRESSION):
        self.model = model
        self.X_train = X_train
        self.y_train = self._check_input(y_train)
        self.y_pred = self._check_input(y_pred) if y_pred is not None else None
        self.mode: str = mode

    
    def _check_input(self, data):
        """
        Ensures the input data is in a consistent format.
        """
        if isinstance(data, pd.Series):
            return data.to_frame()
        elif isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        return data

    @abstractmethod
    def explain_global(self, x_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates a global explanation of the model predictions over the entire dataset.

        Parameters:
            x_data (pd.DataFrame): Dataset for which the global explanation is required.

        Returns:
            pd.DataFrame: A DataFrame containing the global explanation results.
        """
        pass

    @abstractmethod
    def explain_local(self, x_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates a local explanation of the model predictions for individual samples.

        Parameters:
            x_data (pd.DataFrame): Dataset for which local explanations are required.

        Returns:
            pd.DataFrame: A DataFrame containing the local explanation results for each sample.
        """
        pass
