# ----------------------------------------------------------------------------------------------------
# Main file for testing
#
#
# ------------------------------------------------------------------------------------------------------

# Standard library imports
import pandas as pd
import numpy as np

# Third party imports
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Local application imports
from ExplainerFactory import ExplainerFactory


# Use a dataset from the sklearn library for housing prices
dataset_1 = fetch_california_housing(as_frame=True)
X = dataset_1['data']
y = dataset_1['target']

# Cut dataset for processing speed goal
X_small = X[:500]
y_small = y[:500]

# Splits into X and Y first and then splits it further into validation data and training data
X_train, X_test, y_train, y_test = train_test_split(X_small, y_small, test_size=0.2)

'''
# Initialize the scaler
scaler = StandardScaler()

# Normalize the training data
train_X_scaled = scaler.fit_transform(X_train)

# Normalize the validation data using the same scaler
val_X_scaled = scaler.transform(X_test)

# Convert scaled data back to DataFrame
X_train = pd.DataFrame(train_X_scaled, columns=X_train.columns)
X_test = pd.DataFrame(val_X_scaled, columns=X_test.columns)
'''

# Create a random forest model then train it
my_model = RandomForestRegressor(random_state=0)
my_model.fit(X_train, y_train)
print('\n Model trained')

# Initialize the ExplainerFactory with the trained model and data splits.
# This factory is responsible for creating specific explainers as requested.
expl_fctry = ExplainerFactory(my_model, X_train, X_test, y_train, y_test)

# Create a LIME explainer using the factory and get global explanations. LIME is used for 
# explaining predictions by approximating the model locally around the prediction.
limeEx_f = expl_fctry.create_explainer('lime')
lime_global = limeEx_f.explain_global(X_train)
print('\n Lime explanation created')

# Create a SHAP explainer and get global explanations. SHAP values explain the impact of having a certain value
# for a given feature in comparison to the prediction we'd make if that feature took some baseline value.
shapEx_f = expl_fctry.create_explainer('shap')
shap_global = shapEx_f.explain_global(X_train)
print('\n SHAP explanation created')

results = pd.concat([lime_global, shap_global], axis=1)

def plot_lime_shap(data, shap_column, lime_column):
    
    colors = sns.color_palette("deep")
    plt.figure(figsize=(8, 6))    
    
    bar_positions = np.arange(len(data))  # Positions of the bars
    bar_width = 0.35  # Bar widths

    plt.barh(bar_positions - bar_width/2, data[shap_column], height=bar_width, label='SHAP', color=colors[0])  # PSHAP values
    plt.barh(bar_positions + bar_width/2, data[lime_column], height=bar_width, label='LIME', color=colors[1])  # LIME values
    plt.yticks(bar_positions, data.index)  #labels

    plt.title('Feature Importances from SHAP and LIME')
    plt.legend()
    plt.show()

plot_lime_shap(results, 'SHAP Value', 'LIME Value')
print('\n plotting completed')