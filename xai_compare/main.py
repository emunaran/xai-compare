"""
Main File for Testing the XAI Comparison Package

This script demonstrates the setup and usage of the XAI Comparison package,
using a RandomForestClassifier on the German credit dataset.

Imports:
    - RandomForestClassifier from sklearn.ensemble
    - ComparisonFactory from xai_compare.factory
    - german_credit dataset from xai_compare.datasets
    - MODE from xai_compare.config

Steps:
    1. Define the model
    2. Load the dataset
    3. Set the mode (classification/regression)
    4. Configure parameters
    5. Initialize the ComparisonFactory
    6. Create and apply comparisons
"""


# Standard library imports
from sklearn.ensemble import RandomForestClassifier

# Local application imports
from xai_compare.factories.factory import ComparisonFactory
from xai_compare.datasets import german_credit
from xai_compare.config import MODE, COMPARISON_TECHNIQUES

if __name__ == '__main__':

    # Step 1: Define the model
    model = RandomForestClassifier(max_depth=4, n_estimators=100, random_state=42)

    # Step 2: Load the dataset
    X, y = german_credit()

    # Step 3: Set the mode
    mode = MODE.CLASSIFICATION

    # Step 4: Configure parameters
    params = {'model': model,
              'data': X,
              'target': y,
              'custom_explainer': None,
              'verbose': False,
              'mode': mode,
              # 'default_explainers': ['shap', 'lime', 'permutations']  # Uncomment to specify explainers
              }

    # Option to create a custom explainer (uncomment and define your custom explainer)
    # my_custom_explainer = create_my_custom_explainer(...)

    # Step 5: Initialize the ComparisonFactory
    # The ComparisonFactory is responsible for creating comparison objects based on the provided parameters.
    comparison_factory = ComparisonFactory(**params)

    # Step 6: Create and apply comparisons
    for technique in COMPARISON_TECHNIQUES:
        comparison = comparison_factory.create(technique)
        comparison.apply()
        comparison.display()
