# ----------------------------------------------------------------------------------------------------
# Module explainer_utilities
#
# This module contains utility functions for working with various explanation methods.
# The functions are designed to facilitate the generation, collection, and handling of explanations from
# different explainers.
# ------------------------------------------------------------------------------------------------------

import pandas as pd


def run_and_collect_explanations(explainer, X_data, verbose=True) -> pd.DataFrame:
    """
    Executes global explanation methods provided by the specified explainer on the given dataset, collecting the results.

    Attributes:
        explainer:
            An explainer object that must have an `explain_global` method.

        X_data (pd.DataFrame):
            The dataset for which explanations are to be generated.

        verbose (bool, optional):
            If True, prints out the status of explanation generation. Default is True.

    Returns:
        pd.DataFrame:
            A DataFrame containing the global explanations for each feature in the dataset. If the explanation generation
            fails for all features or is not attempted, returns an empty DataFrame.

    Raises:
        Exception:
            Captures and prints any exceptions raised during the explanation generation, indicating failure.
    """
    
    try:
        global_explanation = explainer.explain_global(X_data)
        if verbose:
            print(f'\n {explainer.__name__} explanation created')
        return global_explanation
    except Exception as e:
        print(f"Failed to create {explainer.__name__} explanation: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if explanation generation fails
