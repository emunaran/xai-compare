# ----------------------------------------------------------------------------------------------------
# Module Comparison
#
# This module contains classes for comparing model Explainers, evaluating their consistency, and assessing 
# feature selection strategies. It provides a framework for generating comparison reports, visualizing results, 
# and measuring the stability and performance of different explainer methods on machine learning models.
#
# ------------------------------------------------------------------------------------------------------

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Union, List, Type

# Local application imports
from xai_compare.config import MODE, EXPLAINERS
from xai_compare.factories.factory import ExplainerFactory
from xai_compare.abstract.explainer import Explainer


class Comparison(ABC):
    """
    Base class for model comparison that handles various explainer analyses.

    This abstract class provides a framework for comparing explanation methods
    to assess feature importance and explainer consistency.

    Attributes:
        model (Model):
            The input machine learning model.

        data (pd.DataFrame):
            The feature dataset used for model training and explanation.

        target (Union[pd.DataFrame, pd.Series, np.ndarray]):
            The target variables associated with the data.

        mode (str, default 'REGRESSION'):
            The mode of operation from config.py.

        random_state (int, default 42):
            Seed used by the random number generator for reproducibility.

        verbose (bool, default True):
            Enables verbose output during operations.

        default_explainers (List[str], default EXPLAINERS):
            List of default explainers from config.py.

        custom_explainer (Union[Type[Explainer], List[Type[Explainer]], None], optional):
            Custom explainer classes to be added to the default explainers.

    Methods:
        apply():
            Abstract method to generate a comparison report based on the explainer outputs.

        display():
            Abstract method to plot and display the result from the comparison analysis.
    """

    def __init__(self,
                 model,
                 data: pd.DataFrame,
                 target: Union[pd.DataFrame, pd.Series, np.ndarray],
                 mode: str = MODE.REGRESSION, 
                 random_state: int = 42, 
                 verbose: bool = True,
                 default_explainers: List[str] = EXPLAINERS,
                 custom_explainer: Union[Type[Explainer], List[Type[Explainer]], None] = None):

        if not hasattr(model, 'fit'):
            raise ValueError("The model should have a 'fit' method.")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data should be a pandas DataFrame.")
        if not isinstance(target, (pd.DataFrame, pd.Series, np.ndarray)):
            raise TypeError("Target should be a pandas DataFrame, Series, or a numpy ndarray.")
        if mode not in [MODE.REGRESSION, MODE.CLASSIFICATION]:
            raise ValueError(f"Invalid mode. Expected 'REGRESSION' or 'CLASSIFICATION', got {mode}.")
        if not isinstance(random_state, int):
            raise TypeError("Random state should be an integer.")
        if not isinstance(verbose, bool):
            raise TypeError("Verbose should be a boolean.")
        if not isinstance(default_explainers, list):
            raise TypeError("Default explainers should be a list of strings.")
        if not all(explainer in EXPLAINERS for explainer in default_explainers):
            raise ValueError(f"Some default explainers are not in the allowed EXPLAINERS list: {default_explainers}")
        if custom_explainer and not (
                isinstance(custom_explainer, Explainer) or
                issubclass(custom_explainer, Explainer) or
                isinstance(custom_explainer, list) and all(isinstance(e, Explainer) for e in custom_explainer) or
                custom_explainer is None):
            raise TypeError("Custom explainer should be an Explainer type, a list of Explainer types, or None.")

        self.model = model
        self.data = data
        self.y = target
        self.mode = mode
        self.random_state = random_state
        self.verbose = verbose
        self.default_explainers = default_explainers
        self.list_explainers = self.create_list_explainers(custom_explainer)

    def create_list_explainers(self, custom_explainer: Union[Type[Explainer], List[Type[Explainer]], None]) -> List[Explainer]:
        """
        Creates a list of explainer classes from default and custom explainers.

        Attributes:
            custom_explainer (Union[Type[Explainer], List[Type[Explainer]], None]):
                Custom explainer or a list of custom explainer classes.

        Returns:
            List[Explainer]:
                A list of initialized explainer classes.
        """

        list_explainers = [ExplainerFactory().create(explainer_name) for explainer_name in self.default_explainers]

        if custom_explainer:
            # Ensure custom_explainer is a list, even if it's a single object
            custom_explainer = custom_explainer if isinstance(custom_explainer, list) else [custom_explainer]
            list_explainers.extend(custom_explainer)

        if not list_explainers:
            raise ValueError("The list of explainers is empty. Please provide valid default or custom explainers.")

        return list_explainers

    @abstractmethod
    def apply(self):
        pass

    @abstractmethod
    def display(self):
        pass
