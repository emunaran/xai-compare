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
                 train_x: pd.DataFrame,
                 train_y: pd.DataFrame,
                 y_pred: pd.DataFrame):
        self.model = model
        self.train_x = train_x
        self.train_y: pd.DataFrame = train_y
        self.y_pred: pd.DataFrame = y_pred

    @abstractmethod
    def explain_global(self, x_data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def explain_local(self,
                      x_data: [pd.DataFrame]) -> pd.DataFrame:
        pass
