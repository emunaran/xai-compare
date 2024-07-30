# ----------------------------------------------------------------------------------------------------
# Class ExplainerFactory
# This class is used to create Explainer objects based on what word is entered into the create() function
#
# ------------------------------------------------------------------------------------------------------

import string
from typing import Any, Union, List, Type

import pandas as pd
import numpy as np

# Local application imports
from xai_compare.abstract.explainer import Explainer
from xai_compare.explainers.lime_wrapper import LIME
from xai_compare.explainers.shap_wrapper import SHAP
from xai_compare.explainers.permutation_wrapper import Permutations
from xai_compare.config import MODE, EXPLAINERS


class ExplainerFactory:
    """
    A class for creating instances of various explainer types based on the given model and data.

    This class simplifies the process of instantiating different types of explainers by providing a common interface
    to specify the necessary training and testing datasets along with the model.

    Attributes:
        model (Any):
            The machine learning model to be explained.

        X_train (pd.DataFrame):
            Training features dataset.

        X_test (pd.DataFrame):
            Testing features dataset.

        y_train (Union[pd.DataFrame, pd.Series, np.ndarray]):
            Training target dataset.

        y_test (Union[pd.DataFrame, pd.Series, np.ndarray]):
            Testing target dataset.

        mode (str):
            Mode of operation, specifies if the model is for 'regression' or 'classification'.
    """

    def __init__(self,
                 model: Any = None,
                 X_train: pd.DataFrame = None,
                 X_test: pd.DataFrame = None,
                 y_train: Union[pd.DataFrame, pd.Series, np.ndarray] = None,
                 y_test: Union[pd.DataFrame, pd.Series, np.ndarray] = None,
                 mode: str = MODE.REGRESSION):
        
        if model and not hasattr(model, 'fit'):
            raise ValueError("The model should have a 'fit' method.")
        if X_train is not None and not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train should be a pandas DataFrame.")
        if X_test is not None and not isinstance(X_test, pd.DataFrame):
            raise TypeError("X_test should be a pandas DataFrame.")
        if y_train is not None and not isinstance(y_train, pd.DataFrame):
            raise TypeError("y_train should be a pandas DataFrame.")
        if y_test is not None and not isinstance(y_test, pd.DataFrame):
            raise TypeError("y_test should be a pandas DataFrame.")
        if mode not in [MODE.REGRESSION, MODE.CLASSIFICATION]:
            raise ValueError(f"Invalid mode. Expected 'REGRESSION' or 'CLASSIFICATION', got {mode}.")
        
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

        Attributes:
            explainer_type (str):
                A string identifier for the explainer type. Valid options are "shap", "lime", "permutations".

        Returns:
            Explainer:
                An instance of the requested explainer type with the provided model and data.

        Raises:
            ValueError:
                If the explainer_type is not recognized or unsupported.
        """

        if explainer_type == "shap":
            return SHAP(self.model, self.X_train, self.y_train) if self.model else SHAP
        elif explainer_type == "lime":
            return LIME(self.model, self.X_train, self.y_train) if self.model else LIME
        elif explainer_type == "permutations":
            return Permutations(self.model, self.X_train, self.y_train) if self.model else Permutations
        else:
            raise ValueError("Invalid Explainer type provided")

        # If there are more explainers you want to account for, the code can be added here


# ----------------------------------------------------------------------------------------------------
# Class ComparisonFactory
# This class is used to create Comparison objects based on the name of comparison technique and explainers
#
# ------------------------------------------------------------------------------------------------------

# Local application imports

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
                 mode: str = MODE.REGRESSION, 
                 random_state: int = 42, 
                 verbose: bool = True,
                 threshold: float = 0.2,
                 metric: Union[str, None] = None,
                 n_splits: int = 5,
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
        if not isinstance(threshold, (int, float)) or not (0 < threshold <= 1):
            raise ValueError("Threshold should be a float between 0 and 1.")
        if metric and metric not in ['Accuracy', 'Precision', 'Recall', 'F1_score', 'AUC', 'MSE', 'MAE']:
            raise ValueError(f"Invalid metric '{metric}'. Valid options are 'Accuracy', 'Precision', 'Recall', 'F1_score', 'AUC', 'MSE', 'MAE'.")
        if not isinstance(n_splits, int) or n_splits <= 1:
            raise ValueError("n_splits should be an integer greater than 1.")
        if not isinstance(default_explainers, list):
            raise TypeError("Default explainers should be a list of strings.")
        if not all(explainer in EXPLAINERS for explainer in default_explainers):
            raise ValueError(f"Some default explainers are not in the allowed EXPLAINERS list: {default_explainers}")
        
        self.model = model
        self.data = data
        self.y = target
        self.mode = mode
        self.random_state = random_state
        self.verbose = verbose
        self.threshold = threshold
        self.metric = metric
        self.n_splits = n_splits
        self.default_explainers = default_explainers
        self.custom_explainer = custom_explainer

        self.model.fit(self.data, self.y)

    def create(self, comparison_type: string):
        """
        Creates and returns a comparison object based on the specified type.

        This method allows for dynamic creation of different types of comparison objects based on a predefined string
        identifier. If the model is not set, it returns the class of the comparison for manual instantiation.

        Attributes:
            comparison_type (str):
                A string identifier for the comparison type. Valid options are "feature_selection", "consistency".

        Returns:
            comparison.Comparison:
                An instance or class of the requested comparison type with the provided model and data.

        Raises:
            ValueError:
                If the comparison_type is not recognized or unsupported.
        """


        if comparison_type == "feature_selection":
            # Importing locally to avoid circular dependency
            from xai_compare.comparisons import FeatureSelection
            if self.model:
                    feature_selection_comparison = FeatureSelection(model=self.model,
                                                             data=self.data,
                                                             target=self.y,
                                                             custom_explainer=self.custom_explainer,
                                                             mode=self.mode,
                                                             random_state=self.random_state,
                                                             verbose=self.verbose,
                                                             threshold=self.threshold,
                                                             metric=self.metric,
                                                             default_explainers=self.default_explainers)
                    
            else:
                feature_selection_comparison = FeatureSelection
            return feature_selection_comparison
        elif comparison_type == "consistency":
            # Importing locally to avoid circular dependency
            from xai_compare.comparisons import Consistency
            if self.model:
                consistency_comparison = Consistency(model=self.model,
                                              data=self.data,
                                              target=self.y,
                                              custom_explainer=self.custom_explainer,
                                              mode=self.mode,
                                              random_state=self.random_state,
                                              verbose=self.verbose,
                                              n_splits=self.n_splits,
                                              default_explainers=self.default_explainers)
            else:
                consistency_comparison = Consistency
            return consistency_comparison
        
        # If there are more Comparison Techniques you want to account for, the code can be added here:
        
        else:
            raise ValueError("Invalid Comparison Technique provided")    
