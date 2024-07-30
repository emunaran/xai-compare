# ----------------------------------------------------------------------------------------------------
# Class SHAP
# This clas wraps SHAP explainer methods.
#
# ------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import shap as sh

# Local application imports
from xai_compare.abstract.explainer import Explainer


class SHAP(Explainer):
    """
    A class that encapsulates the SHAP (SHapley Additive exPlanations) method for explaining model predictions.
    The method is detailed in the paper "A Unified Approach to Interpreting Model Predictions"
    (https://arxiv.org/pdf/1705.07874).

    Attributes:
        model:
            An input machine learning model.

        mode (str):
            Indicates whether the explainer is used for 'regression' or 'classification'.
    """

    __name__ = "SHAP"

    def explain_global(self, x_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates global SHAP values (average) for the features in the dataset.

        Attributes:
            x_data (pd.DataFrame):
                DataFrame containing the feature data.

        Returns:
            pd.DataFrame:
                DataFrame of average SHAP values for each feature.
        """

        shap_values = self.explain_local(x_data)
        shap_mean = np.mean(shap_values, axis=0)
        return pd.DataFrame(shap_mean, index=x_data.columns, columns=['SHAP Value'])

    def choose_explainer(self, model_type: str) -> sh.Explainer:
        """
        Selects an appropriate SHAP explainer based on the model type.

        Attributes:
            model_type (str):
                A string describing the type of the model.

        Returns:
            sh.Explainer:
                A SHAP Explainer class or None if no appropriate explainer is found.
        """

        model_type = model_type.lower()

        if "tree" in model_type or "forest" in model_type or "xgb" in model_type:
            return sh.TreeExplainer
        elif "linear" in model_type:
            return sh.LinearExplainer
        else:
            return sh.KernelExplainer

    def explain_local(self, x_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates local SHAP values for the given data points.

        Attributes:
            x_data (pd.DataFrame):
                DataFrame containing the feature data.

        Returns:
            pd.DataFrame:
                DataFrame of SHAP values for each feature and data point.
        """

        explainer_class = self.choose_explainer(type(self.model).__name__)

        background = sh.kmeans(x_data, 5).data if explainer_class.__name__ != "LinearExplainer" else x_data

        explainer = explainer_class(self.model, background)

        try:
            shap_values = explainer.shap_values(x_data)
        except:
            shap_values = explainer.shap_values(x_data, check_additivity=False)

        if not isinstance(shap_values, list):
            # Regression task or classification with linear model
            shap_df = pd.DataFrame(shap_values, columns=x_data.columns)

        elif len(shap_values) == 2:
            # Binary classification task
            # In binary classification tasks, SHAP returns two items in the shap_values array.
            # The item at index 0 represents the SHAP values for the negative class (label 0),
            # and the item at index 1 represents the SHAP values for the positive class (label 1).
            # Here, we are taking only the SHAP values for the positive class.
            shap_df = pd.DataFrame(shap_values[1], columns=x_data.columns)

        else:       # Multiclass
            # Stack the arrays and average over classes (axis=-1)
            stacked_shap_values = np.stack(np.abs(shap_values), axis=-1)
            mean_abs_shap_values = np.mean(stacked_shap_values, axis=-1)
            shap_df = pd.DataFrame(mean_abs_shap_values, columns=x_data.columns)

        return shap_df
