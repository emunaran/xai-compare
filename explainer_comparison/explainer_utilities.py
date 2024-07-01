import pandas as pd
from explainer_comparison.ExplainerFactory import ExplainerFactory
from explainer_comparison.config import EXPLAINERS

def run_and_collect_explanations(factory: ExplainerFactory, X_data, explainers=None, verbose=True) -> pd.DataFrame:
    results = []
    available_explainers = EXPLAINERS  # Easily extendable for additional explainers
    
    # chosen_explainers = explainers if explainers is not None else available_explainers

    if explainers is None:
        chosen_explainers = available_explainers
    elif type(explainers)==list:
        chosen_explainers = explainers
    else:
        chosen_explainers = [explainers]

    for explainer_type in chosen_explainers:
        explainer = factory.create_explainer(explainer_type)
        if explainer is not None:
            try:
                global_explanation = explainer.explain_global(X_data)
                results.append(global_explanation)
                if verbose:
                    print(f'\n {explainer_type.upper()} explanation created')
            except Exception as e:
                print(f"Failed to create {explainer_type.upper()} explanation: {e}")
        else:
            print(f"No explainer available for type: {explainer_type}")

    # Concatenate all results along columns (axis=1), handling cases where some explanations might fail
    if results:
        return pd.concat(results, axis=1)
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no explanations were added
