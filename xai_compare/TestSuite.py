from typing import Any
import numpy as np
import pandas as pd
import shap
from sklearn.datasets import fetch_california_housing, load_iris, load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from lime_wrapper import LIME
from shap_wrapper import SHAP

# Local application imports
from xai_compare.explainer import Explainer


# Using 3-4 existing datasets, either from SHAP, sklearn, or pandas is fine.
# Then training 3-4 different models (for each dataset).
# Finally extracting the information from the explainers, local and global.
# Don't worry, at this point, for the correctness of the results.
# We have other tests for that.
# Just make sure that you get valid results (non empty, not zeros).
# You may compare, or plot the results against each other.

def main():
    # first dataset: california housing
    dataset_1 = fetch_california_housing(as_frame=True)
    testing_regressable_dataset(dataset_1)

    # second dataset: iris
    dataset_2 = load_iris()

    # Convert multiclass data to Binary class data
    binary_target = np.where(dataset_2.target == 0, 1, 0)
    # iris_2d = dataset_2.data[:, 2:4]

    X_2 = dataset_2.data
    y_2 = binary_target
    X_2 = X_2[:200]
    y_2 = y_2[:200]
    testing_classifiable_dataset(X_2, y_2)

    # third dataset: diabetes
    X_3, y_3 = load_diabetes(return_X_y=True, as_frame=True)

    # Drop categorical data
    X_3 = X_3.drop(["sex"], axis=1)
    X_3 = X_3[:200]
    y_3 = y_3[:200]

    # fourth dataset: breast cancer
    dataset_4 = load_breast_cancer(as_frame=True)
    X_4 = dataset_4['data']
    y_4 = dataset_4['target']
    X_4 = X_4[:200]
    y_4 = y_4[:200]


# Helper method that creates a Rand. Forest Regressor based on training data and returns a trained model:
def create_RandomForestRegressor(train_X: pd.DataFrame, train_y: pd.DataFrame) -> RandomForestRegressor:
    my_model = RandomForestRegressor(random_state=0)
    my_model.fit(train_X.to_numpy(), train_y.to_numpy())
    return my_model


# Helper method that creates a Rand. Forest Classifier based on training data and returns a trained model:
def create_RandomForestClassifier(train_X: pd.DataFrame, train_y: pd.DataFrame) -> RandomForestClassifier:
    my_model_classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    my_model_classifier.fit(train_X, train_y)
    return my_model_classifier


# Helper method that runs tests for Lime, Shap, and XGBoost for a regressable dataset
def testing_regressable_dataset(dataset: pd.DataFrame) -> Any:
    # Tests Using Dataset

    # Splits dataset into X and Y first(and shortens the data to 200 data pts,
    # then splits data further into validation data and training data
    X = dataset['data']
    y = dataset['target']
    X = X[:200]
    y = y[:200]
    # test size indicates the ratio of the training data pts to the total data pts.
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2)

    # Create a random forest model then train it
    my_model = create_RandomForestRegressor(train_X, train_y)

    # After training, this is the real fit
    y_pred = my_model.predict(val_X)

    # Testing SHAP local explanation

    # creating an explainer to show what the SHAP wrapper class's methods should output
    tree_explainer = shap.TreeExplainer(my_model)

    # initializing all of the SHAP object's fields (setup)
    shapEx = SHAP(my_model, train_X, train_y, y_pred)

    # Testing SHAP.explain_local:
    # check_SHAP_Local(my_model, tree_explainer, shapEx, X, val_X, y_pred)

    # Testing SHAP.explain_global:
    # check_SHAP_Global(my_model, shapEx, val_X)

    # Testing LIME.explain_local:

    limeEx = LIME(my_model, X.to_numpy(), y, my_model.predict(val_X))
    check_LIME_local(limeEx, val_X, y_pred)

    # Testing LIME.explain_global:
    check_LIME_global(limeEx, val_X, y_pred)


# Runs all tests for Lime, Shap, and XGBoost on a classifiable dataset
def testing_classifiable_dataset(X: pd.DataFrame, y: pd.DataFrame) -> Any:
    # splits X and y further into training and validation data

    # test size indicates the ratio of the training data pts to the total data pts.
    val_X: pd.DataFrame
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2)

    # Create a random forest classifier then train it
    my_model = create_RandomForestClassifier(train_X, train_y)

    # After training, this is the real fit
    y_pred = my_model.predict(val_X)

    # Setup for testing Local explanation
    print("Testing SHAP local explanation\n")
    # creating an explainer to show what the SHAP wrapper class's methods should output
    tree_explainer = shap.TreeExplainer(my_model)

    # initializing all of the SHAP object's fields (setup)
    shap_ex = SHAP(my_model, train_X, train_y, y_pred)

    # Testing SHAP.explain_local:
    # check_SHAP_Local(my_model, tree_explainer, shap_ex, X, val_X, y_pred)

    # Testing SHAP.explain_global:
    print("\nTesting SHAP global explanation\n")
    # check_SHAP_Global(my_model, shap_ex, val_X)

    # Testing LIME.explain_local:
    print("\nTesting LIME local explanation\n")
    lime_ex = LIME(my_model, X, y, y_pred)
    check_LIME_local(lime_ex, val_X, y_pred)

    # Testing LIME.explain_global:
    print("\nTesting LIME global explanation\n")
    check_LIME_global(lime_ex, val_X, y_pred)


# Prints the results of using the SHAP wrapper class for the global explanation
def check_SHAP_Global(SHAP_wrapper_class_exp: SHAP,
                      val_X: pd.DataFrame) -> None:
    global_exp = SHAP_wrapper_class_exp.explain_global(val_X)
    print(global_exp)


# Prints the results of using a Shap explainer directly for a local Explanation,
# to using the wrapper class SHAP for the local explanation
def check_SHAP_Local(regular_exp,
                     SHAP_wrapper_class_exp: Explainer,
                     X: pd.DataFrame,
                     val_X: pd.DataFrame,
                     y_pred: pd.DataFrame) -> None:
    # expected result:

    # wrapper class's method output:
    if isinstance(SHAP_wrapper_class_exp.model, RandomForestRegressor):
        print("expected output (using RandomForestRegressor):\n")
        print(pd.DataFrame(regular_exp.shap_values(X).std(axis=0)))
    elif isinstance(SHAP_wrapper_class_exp.model, RandomForestClassifier):
        print("expected output (using RandomForestClassifier):\n")
        print(pd.DataFrame(np.array(regular_exp.shap_values(X)[0])))
    print("Output from SHAP wrapper Local explainer method")
    print(SHAP_wrapper_class_exp.explain_local(val_X))


# Helper method that prints the expected output of Lime, and the actual output of the LIME wrapper class
# local explanation method:
def check_LIME_local(LIME_wrapper_class_exp: Explainer,
                     val_X: pd.DataFrame,
                     y_pred: pd.DataFrame) -> None:
    # if isinstance(model, RandomForestRegressor):
    # print("expected output (using RandomForestRegressor):\n")
    # for i in val_X.to_numpy():
    #    print(regular_exp.explain_instance(i, model.predict, num_features=X.columns.size))
    # elif isinstance(model, RandomForestClassifier):
    #     print("expected output (using RandomForestClassifier):\n")
    #     for i in val_X.to_numpy():
    #         print(regular_exp.explain_instance(i, model.predict, num_features=X.columns.size))
    #     print(pd.DataFrame(np.array(regular_exp.shap_values(X)[0])))
    print("Output from LIME wrapper Local explainer method")
    print(LIME_wrapper_class_exp.explain_local(pd.DataFrame(val_X)))


# Helper method that prints the actual output of the LIME wrapper class global explanation method:
def check_LIME_global(LIME_wrapper_class_exp,
                      val_X: pd.DataFrame,
                      y_pred: pd.DataFrame) -> None:
    # if isinstance(model, RandomForestRegressor):
    # print("expected output (using RandomForestRegressor):\n")
    # for i in val_X.to_numpy():
    #    print(regular_exp.explain_instance(i, model.predict, num_features=X.columns.size))
    # elif isinstance(model, RandomForestClassifier):
    #     print("expected output (using RandomForestClassifier):\n")
    #     for i in val_X.to_numpy():
    #         print(regular_exp.explain_instance(i, model.predict, num_features=X.columns.size))
    #     print(pd.DataFrame(np.array(regular_exp.shap_values(X)[0])))
    print("Output from LIME wrapper Global explainer method")
    print(LIME_wrapper_class_exp.explain_global(pd.DataFrame(val_X), y_pred))


if __name__ == '__main__':
    main()
