# ----------------------------------------------------------------------------------------------------
# Class ExplainerFactory
# This class is used to create Explainer objects based on what word is entered into the create() function
#
# ------------------------------------------------------------------------------------------------------

import string
from typing import Any
from typing import Union

import pandas as pd
import numpy as np

# Local application imports
from xai_compare import comparison
from xai_compare.config import MODE, EXPLAINERS, COMPARISON_TECHNIQUES


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



    def create_comparison(self, comparison_type: string) -> comparison.Comparison:
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
            if self.model:
                feature_selectionTchn = comparison.FeatureElimination(model = self.model,
                                                         data = self.data,
                                                         target = self.y,
                                                         custom_explainer = self.custom_explainer,
                                                         mode = self.mode, 
                                                         random_state = self.random_state, 
                                                         verbose=self.verbose,
                                                         threshold=self.threshold,
                                                         metric=self.metric)
            else:
                feature_selectionTchn = comparison.FeatureElimination
            return feature_selectionTchn
        
        elif comparison_type == "consistency":
            if self.model:
                consistencyTchn = comparison.Consistency(model = self.model,
                                                         data = self.data,
                                                         target = self.y,
                                                         custom_explainer = self.custom_explainer,
                                                         mode = self.mode, 
                                                         random_state = self.random_state, 
                                                         verbose=self.verbose,
                                                         n_splits = self.n_splits,
                                                         default_explainers=self.default_explainers)
            else:
                consistencyTchn = comparison.Consistency
            return consistencyTchn



        # If there are more explainers you want to account for, the code can be added here:

        else:
            # throw exception
            print("invalid Comparison Technique")
