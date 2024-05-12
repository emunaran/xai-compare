# ----------------------------------------------------------------------------------------------------
# Class SHAP
# This clas wraps SHAP explainer methods.
#
# ------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import shap as sh
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from Explainer import Explainer


class SHAP(Explainer):
    # initialize with void values

    def explain_global(self, x_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates global SHAP values (average) for the features in the dataset.

        :param x_data: DataFrame containing the feature data.
        :return: DataFrame of average SHAP values for each feature.
        """
        explainer_class = self.chooseExplainer(type(self.model).__name__)

        # build an Exact explainer and explain the model predictions on the given dataset
        explainer = explainer_class(self.model, self.X_train)
        shap_values = explainer.shap_values(x_data)

        #if isinstance(self.model, RandomForestClassifier):
        #    global_exp = pd.DataFrame(explainer.shap_values(x_data).mean(axis=1))
        #    feature_importance = pd.DataFrame(abs(explainer.shap_values(x_data)).mean(axis=1))
        #    print("Global Explanation:\n")
        #    print(global_exp)
        #    print("Feature Importance:\n")
        #    print(feature_importance)
        #elif isinstance(self.model, RandomForestRegressor):

        shap_mean = np.mean(shap_values, axis=0)

        return pd.DataFrame(shap_mean, index=x_data.columns, columns=['SHAP Value'])


    def chooseExplainer(self, model_type: str) -> sh.Explainer:
        """
        Selects an appropriate SHAP explainer based on the model type.

        :param model_type: A string describing the type of the model
        :return: A SHAP Explainer class or None if no appropriate explainer is found
        """
        if "Tree" in model_type or "Forest" in model_type:
            return sh.TreeExplainer
        elif "linear" in model_type:
            return sh.LinearExplainer
        else:
            return sh.KernelExplainer

    def explain_local(self, x_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates local SHAP values for the given data points.

        :param x_data: DataFrame containing the feature data.
        :return: DataFrame of SHAP values for each feature and data point.
        """
        explainer_class = self.chooseExplainer(type(self.model).__name__)
        explainer = explainer_class(self.model, self.X_train)
        shap_values = explainer.shap_values(x_data)
        return pd.DataFrame(shap_values, columns=x_data.columns)
