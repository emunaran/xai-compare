# ----------------------------------------------------------------------------------------------------
# Class ExplainerFactory
# This class is used to create Explainer objects based on what word is entered into the create() function
#
# ------------------------------------------------------------------------------------------------------

import string
from typing import Any

import pandas as pd

# Local application imports
from xai_compare.explainer import Explainer
from xai_compare.explainers.lime_wrapper import LIME
from xai_compare.explainers.shap_wrapper import SHAP
from xai_compare.explainers.permutation_wrapper import PermutationFeatureImportance
from xai_compare.config import MODE, EXPLAINERS


class ExplainerFactory:
    """
    A class for creating classes or instances of various explainer types based on the given model and data.

    This class simplifies the process of instantiating different types of explainers by providing a common interface
    to specify the necessary training and testing datasets along with the model.

    Attributes:
        model (Any): The machine learning model to be explained.
        X_train (pd.DataFrame): Training features dataset.
        X_test (pd.DataFrame): Testing features dataset.
        y_train (pd.DataFrame): Training target dataset.
        y_test (pd.DataFrame): Testing target dataset.
        mode (str): Mode of operation, specifies if the model is for 'regression' or 'classification'.
    """
    def __init__(self,
                 model: Any = None,
                 X_train: pd.DataFrame = None,
                 X_test: pd.DataFrame = None,
                 y_train: pd.DataFrame = None,
                 y_test: pd.DataFrame = None,
                 mode: str = MODE.REGRESSION):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.mode = mode


    def create_explainer(self, explainer_type: string) -> Explainer:
        """
        Creates and returns an explainer object based on the specified type.

        This method allows for dynamic creation of different types of explainer objects based on a predefined string
        identifier. If the model is not set, it returns the class of the explainer for manual instantiation.

        Parameters:
            explainer_type (str): A string identifier for the explainer type. Valid options are "shap", "lime", "permutations".

        Returns:
            Explainer: An instance of the requested explainer type with the provided model and data.

        Raises:
            ValueError: If the explainer_type is not recognized or unsupported.
        """
        if explainer_type == "shap":
            if self.model:
                shapEx = SHAP(self.model, self.X_train, self.y_train)
            else:
                shapEx = SHAP
            return shapEx
        elif explainer_type == "lime":
            if self.model:
                limeEx = LIME(self.model, self.X_train, self.y_train)
            else:
                limeEx = LIME
            return limeEx
        elif explainer_type == "permutations":
            if self.model:
                permEx = PermutationFeatureImportance(self.model, self.X_train, self.y_train)
            else:
                permEx = PermutationFeatureImportance
            return permEx


        # If there are more explainers you want to account for, the code can be added here:

        else:
            # throw exception
            print("invalid Explainer")
