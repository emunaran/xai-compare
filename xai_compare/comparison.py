# ----------------------------------------------------------------------------------------------------
# Class Comparison
# This is the abstract class with explain_global adn explain_local as the abstract methods
# SHAP, LIME and XGBOOST inherit from this class
# ------------------------------------------------------------------------------------------------------
from abc import ABC, abstractmethod
import pandas as pd
from typing import Union
import numpy as np

# Local application imports
from xai_compare.config import MODE, EXPLAINERS
from xai_compare.explainer_factory import ExplainerFactory


class Comparison(ABC):
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

        list_explainers = [ExplainerFactory().create_explainer(explainer_name) for explainer_name in self.default_explainers]

        if custom_explainer:
            list_explainers.extend(custom_explainer)
        
        return list_explainers

    @abstractmethod
    def comparison_report(self):
        pass

    @abstractmethod
    def best_result(self):
        pass
