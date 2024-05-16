# ----------------------------------------------------------------------------------------------------
# Class Explainer
# This is the abstract class with explain_global adn explain_local as the abstract methods
# SHAP, LIME and XGBOOST inherit from this class
# ------------------------------------------------------------------------------------------------------
from abc import ABC, abstractmethod
import pandas as pd


class Explainer(ABC):
    def __init__(self,
                 model,
                 X_train: pd.DataFrame,
                 y_train: pd.DataFrame,
                 y_pred: pd.DataFrame = None):
        self.model = model
        self.X_train = X_train
        self.y_train: pd.DataFrame = y_train
        self.y_pred: pd.DataFrame = y_pred

    @abstractmethod
    def explain_global(self, x_data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def explain_local(self, x_data: pd.DataFrame) -> pd.DataFrame:
        pass
