# ----------------------------------------------------------------------------------------------------
# Class ExplainerFactory
# This class is used to create Explainer objects based on what word is entered into the create() function
#
# ------------------------------------------------------------------------------------------------------

import string
from typing import Any

import pandas as pd
import xgboost as xgb

import Explainer
from LIME import LIME
from SHAP import SHAP


class ExplainerFactory:
    # If user wants to use XGBoost explanation, model, X, and y must be filled in as parameters for init
    # all possible parameters are set to None as default.
    def __init__(self,
                 model: Any = None,
                 X: pd.DataFrame = None,
                 y: pd.DataFrame = None,
                 val_X: pd.DataFrame = None,
                 train_X: pd.DataFrame = None,
                 val_y: pd.DataFrame = None,
                 train_y: pd.DataFrame = None):
        self.model = model
        self.X = X
        self.y = y
        self.val_X = val_X
        self.val_y = val_y
        self.train_X = train_X
        self.train_y = train_y
        self.__init__()

    def create_explainer(self, explainer_type: string) -> Explainer:
        if explainer_type == "shap":
            shapEx = SHAP(self.model)
            return shapEx
        elif explainer_type == "lime":
            limeEx = LIME(self.model)
            return limeEx
        elif explainer_type == "xgboost":
            return self.create_xgb_global_feature_importance(self.model, self.X, self.y)

        # If there are more explainers you want to account for, the code can be added here:

        else:
            # throw exception
            print("invalid Explainer")

    # Check to see if you can restrict the type of the model and output.
    def create_xgb_global_feature_importance(self,
                                             model: Any,
                                             X: pd.DataFrame,
                                             y: pd.DataFrame) -> Any:
        if not isinstance(model, xgb.XGBClassifier):
            raise ValueError("model must be an XGBoost model")
        elif not isinstance(model, xgb.XGBRegressor):
            raise ValueError("model must be an XGBoost model")
        model.fit(X, y)
        return model.feature_importances_
