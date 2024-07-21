# ----------------------------------------------------------------------------------------------------
# Module datasets
#
# This module provides utility functions for loading and preprocessing datasets. It includes functions
# to fetch data from various sources, normalize datasets, and return features and target labels as 
# pandas DataFrames.
# ------------------------------------------------------------------------------------------------------


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_california_housing
import pkg_resources
from typing import Tuple

# github_data_url = "https://github.com/emunaran/xai-compare/tree/main/data"

def get_data_path(filename: str) -> str:
    """
    Get the full path to a data file included in the package.

    Parameters:
    ----------
    filename : str
        The name of the file.

    Returns:
    -------
    str
        The full path to the data file.
    """
    return pkg_resources.resource_filename('xai_compare', filename)

def german_credit() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Return the normalized German credit score dataset.

    The original dataset is located at: https://online.stat.psu.edu/stat508/resource/analysis/gcd

    Returns:
    -------
    Tuple[pd.DataFrame, pd.Series]
        A tuple containing the features DataFrame and the target Series.

    Feature Columns:
        - 'Account Balance'
        - 'Duration of Credit (month)'
        - 'Payment Status of Previous Credit'
        - 'Purpose'
        - 'Credit Amount'
        - 'Value Savings/Stocks'
        - 'Length of current employment'
        - 'Instalment per cent'
        - 'Sex & Marital Status'
        - 'Guarantors'
        - 'Duration in Current address'
        - 'Most valuable available asset'
        - 'Age (years)'
        - 'Concurrent Credits'
        - 'Type of apartment'
        - 'No of Credits at this Bank'
        - 'Occupation'
        - 'No of dependents'
        - 'Telephone'
        - 'Foreign Worker'

    Target:
        - 'Creditability'

    Examples:
    --------
    To get the processed data and target labels::

        data, target = xai_compare.datasets.german_credit()
    """
    dataset_path = get_data_path('data/german_credit_score/german_credit.csv')

    # Read the dataset
    df = pd.read_csv(dataset_path)
    
    X = df.iloc[:, 1:]
    y = df.Creditability

    # Normalize the data
    normalized_df = MinMaxScaler().fit_transform(X)
    X = pd.DataFrame(normalized_df, columns=X.columns)

    return X, y

def diabetes() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Return the diabetes dataset.

    The original dataset is located at: https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset

    Returns:
    -------
    Tuple[pd.DataFrame, pd.Series]
        A tuple containing the features DataFrame and the target Series.

    Feature Columns:
        - Pregnancies: Number of pregnancies
        - Glucose: Glucose level in blood
        - BloodPressure: Blood pressure measurement
        - SkinThickness: Thickness of the skin
        - Insulin: Insulin level in blood
        - BMI: Body mass index
        - DiabetesPedigreeFunction: Diabetes percentage
        - Age: Age

    Target:
        - Outcome: The final result (1 is Yes, 0 is No)

    Examples:
    --------
    To get the processed data and target labels::

        data, target = xai_compare.datasets.diabetes()
    """
    dataset_path = get_data_path('data/diabetes/diabetes.csv')

    # Read the dataset
    df = pd.read_csv(dataset_path)

    X = df.iloc[:, :-1]
    y = df.Outcome

    return X, y

def california_housing() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Return the California housing dataset from sklearn.datasets.

    This dataset is used for regression tasks.

    The original dataset is located at: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

    Returns:
    -------
    Tuple[pd.DataFrame, pd.Series]
        A tuple containing the features DataFrame and the target Series.

    Feature Columns:
        - MedInc: Median income in block group
        - HouseAge: Median house age in block group
        - AveRooms: Average number of rooms per household
        - AveBedrms: Average number of bedrooms per household
        - Population: Block group population
        - AveOccup: Average number of household members
        - Latitude: Block group latitude
        - Longitude: Block group longitude

    Target:
        - MedHouseVal: Median house value for California districts, expressed in hundreds of thousands of dollars ($100,000).

    Examples:
    --------
    To get the processed data and target labels::

        data, target = xai_compare.datasets.california_housing()
    """
    df = fetch_california_housing(as_frame=True)
    X = df['data']
    y = df['target']

    return X, y
