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


def fico() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Return the preprocessed and normalized FICO dataset.

    The original dataset is located at: https://www.kaggle.com/datasets/parisrohan/credit-score-classification

    To replicate the data preprocessing from raw data to the processed dataset "fico_preprocessed.csv," 
    you can run the notebook located here: xai_compare/data/fico/preprocessed/preprocessing.ipynb

    Returns:
    -------
    Tuple[pd.DataFrame, pd.Series]
        A tuple containing the features DataFrame and the target Series.

    Feature Columns:
        - 'Month': Month of the data record
        - 'Age': Age of the customer
        - 'Occupation': Customer's occupation or job title
        - 'Annual_Income': Annual income of the customer
        - 'Monthly_Inhand_Salary': Net monthly income available to the customer
        - 'Num_Bank_Accounts': Number of bank accounts the customer holds
        - 'Num_Credit_Card': Number of credit cards owned by the customer
        - 'Interest_Rate': Interest rate associated with financial transactions
        - 'Num_of_Loan': Number of loans the customer has
        - 'Type_of_Loan': Type or category of the loan
        - 'Delay_from_due_date': Delay in payment from the due date
        - 'Num_of_Delayed_Payment': Number of delayed payments
        - 'Changed_Credit_Limit': Any recent changes in the customer's credit limit
        - 'Num_Credit_Inquiries': Number of credit inquiries made by the customer
        - 'Credit_Mix': Variety of credit types in the customer's financial profile
        - 'Outstanding_Debt': Total outstanding debt of the customer
        - 'Credit_Utilization_Ratio': Ratio of credit used to credit available
        - 'Credit_History_Age': Age of the customer's credit history
        - 'Payment_of_Min_Amount': Payment behavior regarding the minimum amount due
        - 'Total_EMI_per_month': Total Equated Monthly Installments paid by the customer
        - 'Amount_invested_monthly': Amount invested by the customer monthly
        - 'Payment_Behaviour': General behavior regarding payments
        - 'Monthly_Balance': Monthly balance in the customer's financial accounts

    Target:
        - 'Credit_Score': Numerical representation of the customer's creditworthiness

    Examples:
    --------
    To get the processed data and target labels::

        data, target = xai_compare.datasets.fico()
    """
    dataset_path = get_data_path('data/fico/preprocessed/fico_preprocessed.csv')

    # Read the dataset
    df = pd.read_csv(dataset_path)

    # Extract features and target
    X = df.iloc[:, :-1]
    y = df.Credit_Score

    return X, y
