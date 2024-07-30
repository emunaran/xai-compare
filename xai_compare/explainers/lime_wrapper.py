# ----------------------------------------------------------------------------------------------------
# Class LIME
#
# This class wraps the lime explainer method
#
# ------------------------------------------------------------------------------------------------------
import lime
import numpy as np
import pandas as pd

# Local application imports
from xai_compare.abstract.explainer import Explainer
from xai_compare.config import MODE

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")


class LIME(Explainer):
    """
    A class that encapsulates the LIME (Local Interpretable Model-agnostic Explanations) method for explaining model predictions.

    The method is detailed in the paper "Why Should I Trust You? Explaining the Predictions of Any Classifier" (https://arxiv.org/pdf/1602.04938).

    Attributes:
        model:
            The machine learning model to be explained.

        mode (str):
            Indicates whether the explainer is used for 'regression' or 'classification'.
    """

    __name__ = "LIME"

    def explain_global(self, X_data: pd.DataFrame) -> pd.DataFrame:
        """
        Provides a global explanation of the model by averaging the local explanations across all instances.

        Attributes:
            X_data (pd.DataFrame):
                The dataset for which global explanations are generated.

        Returns:
            pd.DataFrame:
                A DataFrame containing global importance scores for each feature.
        """
        local_exps = self.explain_local(X_data)
        # Calculate the mean across rows to get the average effect of each feature globally
        global_exp = np.mean(local_exps, axis=0)
        # Transpose and convert to DataFrame to match the requested output format
        return pd.DataFrame(global_exp, index=X_data.columns, columns=['LIME Value'])

    def explain_local(self, X_data: pd.DataFrame) -> pd.DataFrame:
        """
        Provides local explanations for each instance in the dataset, explaining the contribution of each feature to the individual prediction.

        Attributes:
            X_data (pd.DataFrame):
                The dataset for which local explanations are generated.

        Returns:
            pd.DataFrame:
                A DataFrame where each row represents an instance with feature contributions for that specific prediction.
        """

        # Initialize the LIME explainer for tabular data
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_data.values,
            feature_names=X_data.columns.tolist(),
            verbose=False, 
            mode=MODE.REGRESSION
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

            contributions = [(index, contribution) for (index, _), (_, contribution) in zip(exp.local_exp[0], exp.as_list())]
            contributions_sorted_by_feature_order = [expl[1] for expl in sorted(contributions)]
            coefs.append(contributions_sorted_by_feature_order)

        column_names = X_data.columns.tolist()
        local_exp = pd.DataFrame(coefs, columns=column_names)
    
        return local_exp

