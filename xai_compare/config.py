
# ----------------------------------------------------------------------------------------------------
# This section defines constants used for configuring machine learning tasks and explainer methods.
#
# MODE: Specifies the types of machine learning tasks (classification and regression).
# EXPLAINERS: Lists the available explanation methods (SHAP, LIME, and Permutations).
# COMPARISON_TECHNIQUES: Lists the available techniques for comparison (feature selection and consistency).
# ------------------------------------------------------------------------------------------------------
class MODE:
    """ 
    Types of machine learning tasks.
    """
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'


EXPLAINERS = ["shap", "lime", "permutations"]  # List of available explainers

COMPARISON_TECHNIQUES = ["feature_selection", "consistency"]  # List of available comparison techniques
