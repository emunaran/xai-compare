# Constants for explainer

class MODE:
    """ 
    Types of machine learning tasks
    """
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'


EXPLAINERS = ["shap", "lime", "permutations"]  # List of available explainers

COMPARISON_TECHNIQUES =["feature_selection", "consistency"]