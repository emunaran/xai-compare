import pandas as pd

def run_and_collect_explanations(explainer, X_data, verbose=True) -> pd.DataFrame:
    """
    Executes global explanation methods provided by the specified explainer on the given dataset, collecting the results.

    This function is designed to run explanation methods, capture any errors during the execution, and return the
    explanations as a pandas DataFrame. Each column in the DataFrame corresponds to a different type of global explanation
    from the explainer, provided that the explanation does not fail.

    Parameters:
        explainer: An explainer object that must have an `explain_global` method.
        X_data (pd.DataFrame): The dataset for which explanations are to be generated.
        verbose (bool, optional): If True, the function prints out the status of explanation generation. Default is True.

    Returns:
        pd.DataFrame: A DataFrame containing the global explanations for each feature in the dataset. If the explanation
        generation fails for all features or is not attempted, returns an empty DataFrame.

    Raises:
        Exception: Captures and prints any exceptions raised during the explanation generation, indicating failure.
    """
    results = []
    
    try:
        global_explanation = explainer.explain_global(X_data)
        results.append(global_explanation)
        if verbose:
            print(f'\n {explainer.__name__} explanation created')
    except Exception as e:
        print(f"Failed to create {explainer.__name__} explanation: {e}")
 

    # Concatenate all results along columns (axis=1), handling cases where some explanations might fail
    if results:
        return pd.concat(results, axis=1)
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no explanations were added