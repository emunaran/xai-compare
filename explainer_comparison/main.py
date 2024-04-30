# ----------------------------------------------------------------------------------------------------
# Main file for testing
#
#
# ------------------------------------------------------------------------------------------------------
import lime
import shap
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from LIME import LIME
from SHAP import SHAP

# make an Explainer using Explainer Factory
# explainerFact = ExplainerFactory()
# shapExplainer = explainerFact.create_explainer("shap")

# use a dataset from the sklearn library for housing prices
dataset_1 = fetch_california_housing(as_frame=True)
X_1 = dataset_1['data']
y_1 = dataset_1['target']
X_1 = X_1[:200]
y_1 = y_1[:200]

# Splits into X and Y first and then splits it further into validation data and training data
# test size indicates the ratio of the training data pts to the total data pts.
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2)

# Create a random forest model then train it
my_model = RandomForestRegressor(random_state=0)
my_model.fit(train_X.to_numpy(), train_y.to_numpy())
print()
# After training,  this is the real fit
y_pred = my_model.predict(val_X)
print('done')

# creating an explainer to show what the SHAP wrapper class's methods should output
explainer = shap.TreeExplainer(my_model)

# initializing all of the SHAP object's fields (setup)
shapEx = SHAP(my_model, train_X, train_y, y_pred)

# the following print statements are used to compare the wrapper class's method result with the expected result
global_exp = shapEx.explain_global(val_X)
# print(shapEx.explain_local(val_X, y_pred))
# print(explainer.shap_values(X).std(axis=0))
# Lime Tests:

# creating an explainer to show what the LIME wrapper class's methods should output
explainer = lime.lime_tabular.LimeTabularExplainer(train_X.values,
                                                   feature_names=train_X.columns.values.tolist(),
                                                   class_names=['MEDV'], verbose=True, mode='regression')
limeEx = LIME(my_model, train_X, train_y, y_pred)
limeEx.X = X.to_numpy()
limeEx.y = y
y_pred = my_model.predict(val_X.to_numpy())

# the following print statements are used to compare the wrapper class's method result with the expected result
# print(limeEx.explain_local(val_X, val_y))
# X_df = pd.DataFrame(X)
# print("\n")
# for i in val_X.to_numpy():
#    print(explainer.explain_instance(i, my_model.predict, num_features=X_df.columns.size))


#    print("\n")
