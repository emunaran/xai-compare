# ----------------------------------------------------------------------------------------------------
# Class Explainer
# This is the abstract class with explain_global adn explain_local as the abstract methods
# SHAP, LIME and XGBOOST inherit from this class
# ------------------------------------------------------------------------------------------------------
from abc import ABC, abstractmethod
import pandas as pd
from explainer_comparison.config import MODE
from typing import Union
import numpy as np


class Explainer(ABC):
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
        pass

    @abstractmethod
    def explain_local(self, x_data: pd.DataFrame) -> pd.DataFrame:
        pass
