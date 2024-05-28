# ----------------------------------------------------------------------------------------------------
# Class LIME
#
# This class wraps the lime explainer method
#
# ------------------------------------------------------------------------------------------------------
import lime
import numpy as np
import pandas as pd

from explainer_comparison.Explainer import Explainer

# Handle it later
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")


class LIME(Explainer):

    def explain_global(self, X_data: pd.DataFrame) -> pd.DataFrame:
        local_exps = self.explain_local(X_data)
        # Calculate the mean across rows to get the average effect of each feature globally
        global_exp = np.mean(local_exps * X_data.values, axis=0)
        # Transpose and convert to DataFrame to match the requested output format
        return pd.DataFrame(global_exp, index=X_data.columns, columns=['LIME Value'])


    def explain_local(self, X_data: pd.DataFrame) -> pd.DataFrame:
        # Initialize the LIME explainer for tabular data
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_data.values,
            feature_names=X_data.columns.tolist(),
            verbose=False, 
            mode='regression'
        )
        
        # List to store the coefficients for each instance
        coefs = []
        for row in X_data.to_numpy():
            exp = explainer.explain_instance(
                data_row=row, 
                predict_fn=self.model.predict, 
                num_features=len(X_data.columns), 
                num_samples=500
            )

            coeff_for_positive_pred = [expl[1] for expl in sorted(exp.local_exp[1])]
            coeff_for_negative_pred = [expl[1] for expl in sorted(exp.local_exp[0])]
            coeff_ = coeff_for_positive_pred if exp.predicted_value > 0 else coeff_for_negative_pred
            coefs.append(coeff_)

        column_names = X_data.columns.tolist()    
        local_exp = pd.DataFrame(coefs, columns=column_names)
    
        return local_exp

