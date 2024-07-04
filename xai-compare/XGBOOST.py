# ----------------------------------------------------------------------------------------------------
# Class XgBoost
# This class wraps the xgboost explainer methods(both local and global)
#
# ------------------------------------------------------------------------------------------------------
import pandas as pd
import xgboost

from explainer_comparison.Explainer import Explainer


class XGBOOST(Explainer):
    def _init_(self,
               predict_function=None,
               x: [pd.DataFrame] = None,
               y: [pd.DataFrame] = None,
               y_pred: [pd.DataFrame] = None):
        self.predict_function = predict_function
        self.x = x
        self.y = y
        self.y_pred = y_pred

    # lime doesn't work for global explanations
    def explain_global(self,
                       x_data: [pd.DataFrame]) -> pd.DataFrame:
        # Feature_coefficients = None,
        # Feature_ranks = None,
        # Feature_significance = None,
        # Feature_p_values = None)
        data = x_data
        label = x_data.columns
        dtrain = xgboost.DMatrix(data, label=label)
        model = xgboost.training.cv(dtrain=dtrain)
        model.feature_importances()

    def explain_local(self,
                      x_data: [pd.DataFrame]) -> pd.DataFrame:
        # build an XGBOOST explainer and explain the model predictions on the given dataset
        explainer = xgboost.Booster.eval()

        Result = []
        numpy_X = x_data.to_numpy()
        for i in numpy_X:
            e = explainer.explain_instance(i, self.model.predict(x_data),
                                           num_features=x_data.columns)
            Result.append(e)
        # or the predicted y data:
        # exp = explainer.explain_instance(X_data.values, y_predict_data, num_features = X_data.columns.size)
        return Result
