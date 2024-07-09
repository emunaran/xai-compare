import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_california_housing


# github_data_url = "https://github.com/emunaran/xai-compare/tree/main/data"

data_dir = '../../data'


def german_credit():
    """
    Return the normalized german credit score dataset.

    The original dataset is located: https://online.stat.psu.edu/stat508/resource/analysis/gcd

    Returns
    -------
    Pandas DataFrame containing the features and a Pandas Series representing the target.

        Feature Columns:

        'Account Balance', 'Duration of Credit (month)',
        'Payment Status of Previous Credit', 'Purpose', 'Credit Amount',
       'Value Savings/Stocks', 'Length of current employment',
       'Instalment per cent', 'Sex & Marital Status', 'Guarantors',
       'Duration in Current address', 'Most valuable available asset',
       'Age (years)', 'Concurrent Credits', 'Type of apartment',
       'No of Credits at this Bank', 'Occupation', 'No of dependents',
       'Telephone', 'Foreign Worker'

        Target:
        - Creditability

    Examples
    --------
    To get the processed data and target labels::

        data, target = xai_compare.datasets.german_credit()
    """

    # read the dataset
    dataset_dir = f'{data_dir}/german_credit_score/german_credit.csv'

    df = pd.read_csv(dataset_dir)
    
    X = df.iloc[:, 1:]
    y = df.Creditability

    # Normalize the data
    normalized_df = MinMaxScaler().fit_transform(X)
    X = pd.DataFrame(normalized_df, columns=X.columns)

    return X, y



def diabetes():
    """Return the diabetes data.

    The original dataset is located: https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset



    Returns
    -------
    Pandas DataFrame containing the features and a Pandas Series representing the target.

        Feature Columns:

            Pregnancies: To express the Number of pregnancies
            Glucose: To express the Glucose level in blood
            BloodPressure: To express the Blood pressure measurement
            SkinThickness: To express the thickness of the skin
            Insulin: To express the Insulin level in blood
            BMI: To express the Body mass index
            DiabetesPedigreeFunction: To express the Diabetes percentage
            Age: To express the age


        Target:

            Outcome: To express the final result 1 is Yes and 0 is No

    Examples
    --------
    To get the processed data and target labels::

        data, target = xai_compare.datasets.diabetes()

    """

    dataset_dir = f'{data_dir}/diabetes/diabetes.csv'

    df = pd.read_csv(dataset_dir)

    X = df.iloc[:, :-1]
    y = df.Outcome

    return X, y


def california_housing():
    """Return the fetch_california_housing from sklearn.datasets.

    The dataset for the regression mode.

    The original dataset is located: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html



    Returns
    -------
    Pandas DataFrame containing the features and a Pandas Series representing the target.

        Feature Columns:

            MedInc: median income in block group
            HouseAge: median house age in block group
            AveRooms: average number of rooms per household
            AveBedrms: average number of bedrooms per household
            Population: block group population
            AveOccup: average number of household members
            Latitude: block group latitude
            Longitude: block group longitude

        Target:

            MedHouseVal: the median house value for California districts, expressed in hundreds of thousands of dollars ($100,000).

    Examples
    --------
    To get the processed data and target labels::

        data, target = xai_compare.datasets.california_housing()

    """

    df = fetch_california_housing(as_frame=True)
    X = df['data']
    y = df['target']

    return X, y