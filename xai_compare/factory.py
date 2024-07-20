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
from xai_compare.explainers.permutation_wrapper import Permutations
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


    def create(self, explainer_type: string) -> Explainer:
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
                permEx = Permutations(self.model, self.X_train, self.y_train)
            else:
                permEx = Permutations
            return permEx


        # If there are more explainers you want to account for, the code can be added here:

        else:
            # Raise an exception for an invalid explainer type
            raise ValueError("Invalid Explainer type provided")



# ----------------------------------------------------------------------------------------------------
# Class ComparisonFactory
# This class is used to create Comparison objects based on the name of comparison technique and explainers
#
# ------------------------------------------------------------------------------------------------------

import string
from typing import Any
from typing import Union

import pandas as pd
import numpy as np

# Local application imports
from xai_compare.config import MODE, EXPLAINERS


class ComparisonFactory:
    """
    A class for creating classes or instances of various comparison techniques based on the given explainers.

    This class simplifies the process of instantiating different types of comparison techniques by providing 
    a common interface to specify the necessary training and testing datasets along with the model.
    """
    def __init__(self,
                 model,
                 data: pd.DataFrame,
                 target: Union[pd.DataFrame, pd.Series, np.ndarray],
                 custom_explainer = None,
                 mode: str = MODE.REGRESSION, 
                 random_state=42, 
                 verbose=True,
                 threshold=0.2,
                 metric=None,
                 n_splits: int = 5,
                 default_explainers=EXPLAINERS):
    
        
        self.model = model
        self.data = data
        self.y = target
        self.custom_explainer = custom_explainer
        self.mode = mode
        self.random_state = random_state
        self.verbose = verbose
        self.threshold = threshold
        self.metric = metric
        self.n_splits = n_splits
        self.default_explainers = default_explainers

        self.model.fit(self.data, self.y)



    def create(self, comparison_type: string):
        """
        Creates and returns a comparison object based on the specified type.

        This method allows for dynamic creation of different types of explainer objects based on a predefined string
        identifier. If the model is not set, it returns the class of the explainer for manual instantiation.

        Parameters:
            comparison_type (str): A string identifier for the explainer type. 
            Valid options are "feature_selection", "consistency".

        Returns:
            comparison.Comparison: An instance or class of the requested comparison type 
            with the provided model and data.

        Raises:
            ValueError: If the comparison_type is not recognized or unsupported.
        """


        if comparison_type == "feature_selection":
            # Importing locally to avoid circular dependency
            from xai_compare.comparison import FeatureElimination
            if self.model:
                feature_selectionTchn = FeatureElimination(model = self.model,
                                                         data = self.data,
                                                         target = self.y,
                                                         custom_explainer = self.custom_explainer,
                                                         mode = self.mode, 
                                                         random_state = self.random_state, 
                                                         verbose=self.verbose,
                                                         threshold=self.threshold,
                                                         metric=self.metric,
                                                         default_explainers = self.default_explainers)
            else:
                feature_selectionTchn = FeatureElimination
            return feature_selectionTchn
        
        elif comparison_type == "consistency":
            # Importing locally to avoid circular dependency
            from xai_compare.comparison import Consistency
            if self.model:
                consistencyTchn = Consistency(model = self.model,
                                                         data = self.data,
                                                         target = self.y,
                                                         custom_explainer = self.custom_explainer,
                                                         mode = self.mode, 
                                                         random_state = self.random_state, 
                                                         verbose=self.verbose,
                                                         n_splits = self.n_splits,
                                                         default_explainers = self.default_explainers)
            else:
                consistencyTchn = Consistency
            return consistencyTchn


        # If there are more Comparison Techniques you want to account for, the code can be added here:

        else:
            # Raise an exception for an invalid Comparison Technique
            raise ValueError("Invalid Comparison Technique provided")
