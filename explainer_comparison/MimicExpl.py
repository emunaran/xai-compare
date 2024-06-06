# ----------------------------------------------------------------------------------------------------
# Class MimicExpl
#
# This class wraps the interpret.ext.blackbox.MimicExplainer explainer method
#
# ------------------------------------------------------------------------------------------------------
from interpret.ext.blackbox import MimicExplainer

# You can use one of the following four interpretable models as a global surrogate to the black box model
from interpret.ext.glassbox import LGBMExplainableModel
#from interpret.ext.glassbox import LinearExplainableModel
#from interpret.ext.glassbox import SGDExplainableModel
#from interpret.ext.glassbox import DecisionTreeExplainableModel
import numpy as np
import pandas as pd


from explainer_comparison.Explainer import Explainer


class MimicExpl(Explainer):

    def __init__(self, model, X_train, y_train, y_pred=None, mode='regression'):
        super().__init__(model, X_train, y_train, y_pred, mode)
        self._create_explainer()

    def _create_explainer(self):
        self.explainer = MimicExplainer(
            self.model,
            self.X_train,
            LGBMExplainableModel,
            augment_data=True,
            max_num_of_augmentations=10,
            model_task=self.mode,
            classes=list(self.y_train.unique()) if self.mode == 'classification' else None,
            explainable_model_args={
                'objective': 'binary' if self.y_train.nunique().values[0]==2 else self.mode,
                'verbose': -1
            }
        )
    
    def predict(self, X_data: pd.DataFrame) -> pd.DataFrame:
        if self.mode=='classification' and self.y_train.nunique().values[0]==2:
            y_pred_proba = self.predict_proba(X_data)
            y_pred = (y_pred_proba >= 0.5).astype(int)
        else:
            y_pred = self.explainer._get_surrogate_model_predictions(X_data)

        return y_pred
    
    def predict_proba(self, X_data: pd.DataFrame) -> pd.DataFrame:

        if self.mode != 'regression':
            return self.explainer.surrogate_model.predict(X_data)
        
        else:
            raise NotImplementedError('predict_proba is not available for the regression mode')


    def explain_local(self, X_data: pd.DataFrame) -> pd.DataFrame:
        return self.explainer.explain_local(X_data)


    def explain_global(self, X_data: pd.DataFrame) -> pd.DataFrame:
        return self.explainer.explain_global()




