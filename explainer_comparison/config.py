# Constants for explainer

class MODE:
    """ 
    Types of machine learning tasks
    """
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'


EXPLAINERS = ["shap", "lime", "ebm"] #, "mimic"]  # List of available explainers