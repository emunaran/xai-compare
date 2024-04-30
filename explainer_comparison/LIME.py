# ----------------------------------------------------------------------------------------------------
# Class LIME
#
# This class wraps the lime explainer method
#
# ------------------------------------------------------------------------------------------------------
import lime as lime
import numpy as np
import pandas as pd
from numpy import mean

from Explainer import Explainer


class LIME(Explainer):
    # initialize with void values
    def _init_(self,
               model=None,
               x: [pd.DataFrame] = None,
               y: [pd.DataFrame] = None):
        self.model = model
        self.x = x
        self.y = y

        # lime isn't built for global explanations

    def explain_global(self,
                       x_data: pd.DataFrame) -> pd.DataFrame:
        exp = self.explain_local(x_data)
        global_exp = mean(exp, axis=0)
        return global_exp

    def explain_local(self,
                      x_data: [pd.DataFrame]) -> pd.DataFrame:
        explainer = lime.lime_tabular.LimeTabularExplainer(x_data.values,
                                                           feature_names=x_data.columns.values.tolist(),
                                                           class_names=['MEDV'], verbose=False, mode='regression')
        # Explain the prediction using lime's built int explain_instance() function:

        numpy_X = x_data.to_numpy()
        result = []
        for i in numpy_X:
            exp = explainer.explain_instance(i, self.model.predict,
                                             num_features=x_data.columns.size, num_samples=100)
            # Result.append({'coefficients': np.array([e[1] for e in exp.local_exp[0]]),
            #                'intercept': exp.intercept[0]})

            result.append(np.concatenate((np.array([e[1] for e in exp.local_exp[0]]), [exp.intercept[0]])))
        local_exp = pd.DataFrame(data=result)
        # or the predicted y data:
        # exp = explainer.explain_instance(X_data.values, y_predict_data, num_features = X_data.columns.size)
        return local_exp
