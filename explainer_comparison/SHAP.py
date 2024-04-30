# ----------------------------------------------------------------------------------------------------
# Class SHAP
# This clas wraps SHAP explainer methods.
#
# ------------------------------------------------------------------------------------------------------
import pandas as pd
import shap as sh
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from Explainer import Explainer


class SHAP(Explainer):
    # initialize with void values

    def explain_global(self, x_data: pd.DataFrame) -> pd.DataFrame:
        """
        :param x_data: observations to be explained
        """
        # build an Exact explainer and explain the model predictions on the given dataset
        explainer = sh.KernelExplainer(self.model.predict, self.train_x)
        shap_vals = explainer.shap_values(x_data)
        if isinstance(self.model, RandomForestClassifier):
            global_exp = pd.DataFrame(explainer.shap_values(x_data).mean(axis=1))
            feature_importance = pd.DataFrame(abs(explainer.shap_values(x_data)).mean(axis=1))
            print("Global Explanation:\n")
            print(global_exp)
            print("Feature Importance:\n")
            print(feature_importance)
        elif isinstance(self.model, RandomForestRegressor):
            shap_avg = shap_vals.mean(axis=1)
            print(shap_avg)
        # convert explanation into a condensed table of avg. shap vals for each feature.

    # def chooseExplainer(self,
    #                     model_type: string) -> sh.Explainer:
    #     if (model_type.contains("Tree" or "Forest")):
    #         return sh.TreeExplainer
    #     elif (model_type.contains("linear")):
    #         return sh.LinearExplainer
    #     else:
    #         return 0;

    # this method returns an explanation using local data
    def explain_local(self, x_data: pd.DataFrame) -> pd.DataFrame:
        # explainer = sh.TreeExplainer(self.model, x_data)
        explainer = sh.KernelExplainer(self.model.predict, x_data)
        shap_values = pd.DataFrame(explainer.shap_values(x_data))
        # if isinstance(self.model, RandomForestRegressor):
        return shap_values
        # elif isinstance(self.model, RandomForestClassifier):
        #     return shap_values[0]
