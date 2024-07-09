import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold

# Local application imports
from xai_compare.explainer_factory import ExplainerFactory
from xai_compare.explainer_utilities import run_and_collect_explanations

def visualize_consistency(explainers, feature_names, summary):
    """
    Visualizes the mean and standard deviation of feature impacts for different explainers.
    
    Parameters:
        explainers (list): List of explainer names.
        feature_names (list): List of feature names.
        summary (dict): Dictionary containing mean and standard deviation of feature impacts.
    """

    num_explainers = len(explainers)
    fig, axes = plt.subplots(1, num_explainers, figsize=(15, 6), sharey=True)

    # Loop through each explainer and plot the feature impacts
    for ax, explainer in zip(axes, explainers):
        mean_impact, std_impact = summary[explainer]
        ax.barh(feature_names, mean_impact, xerr=std_impact, align='center', alpha=0.7, ecolor='black', capsize=5)
        ax.set_xlabel('Mean Feature Impact')
        ax.set_title(f'Feature Impact and standard deviation - {explainer.upper()}')

    plt.tight_layout()
    plt.show()


def consistency_measurement(X, y, model, n_splits=5, explainers=None, verbose=False, stratified_folds=False):
    """
    Measures the consistency of feature explanations across different folds.
    
    Parameters:
        X (DataFrame): Feature matrix.
        y (Series): Target variable.
        model (sklearn estimator): Machine learning model.
        n_splits (int): Number of splits for cross-validation.
        explainers (list): List of explainer names to use.
        verbose (bool): Verbosity flag.
        stratified_folds (bool): Whether to use StratifiedKFold instead of KFold.
    
    Returns:
        DataFrame: DataFrame containing summary statistics of feature impact standard deviations.
    """

    if stratified_folds:
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    available_explainers = ["shap", "lime"] # Easily extendable for additional explainers
    feature_names = X.columns

    # If no explainers are provided, use all available explainers
    chosen_explainers = explainers if explainers is not None else available_explainers
    results = {explainer: [] for explainer in chosen_explainers}

    # Train the model on the full dataset first
    model.fit(X, y)
    
    # Loop through each fold
    for train_index, test_index in tqdm(folds.split(X, y), total=n_splits, desc="Processing folds"):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Create explainer factory and collect explanations
        factory = ExplainerFactory(model, X_train, X_test, y_train, y_test)
        explanations = run_and_collect_explanations(factory, X_train, explainers=chosen_explainers, verbose=verbose)

        # Store the explanation values for each explainer
        for explainer in chosen_explainers:
            explainer_values = explanations[explainer.upper() + " Value"]
            results[explainer].append(explainer_values)

    # Calculate mean and standard deviation of feature impacts for each explainer
    summary = {}
    for explainer in explainers:
        results[explainer] = np.array(results[explainer])
        mean_impact = np.mean(results[explainer], axis=0)
        std_impact = np.std(results[explainer], axis=0)
        summary[explainer] = (mean_impact, std_impact)

    # Visualize the consistency of feature impacts
    visualize_consistency(chosen_explainers, feature_names, summary)

    # Create a DataFrame summarizing the standard deviation statistics for each explainer
    consistency_scores = {explainer: {'min_std': np.min(summary[explainer][1]), 
                                'max_std': np.max(summary[explainer][1]),
                                'mean_std': np.mean(summary[explainer][1]), 
                                'median_std': np.median(summary[explainer][1])} for explainer in explainers}
    consistency_scores_df = pd.DataFrame(consistency_scores).T
    
    return consistency_scores_df