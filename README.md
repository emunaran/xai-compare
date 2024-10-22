<div align="center">
    <img src="https://raw.githubusercontent.com/emunaran/xai-compare/main/docs/images/xai-compare_logo.png" alt="Logo" width="200"/>
</div>

---
[![PyPI](https://img.shields.io/pypi/v/xai-compare)](https://pypi.org/pypi/xai-compare)
![License](https://img.shields.io/badge/license-MIT-green)
[![Python](https://img.shields.io/badge/python-%3E3.9-blue)](https://pypi.org/project/)
[![Documentation Status](https://readthedocs.org/projects/xai-compare/badge/?version=latest)](https://xai-compare.readthedocs.io/en/latest/?badge=latest)


## Description
`xai-compare` is an open-source library that provides a suite of tools to systematically compare and evaluate the quality of explanations generated by different Explainable AI (XAI) methods. This package facilitates the development of new XAI methods and promotes transparent evaluations of such methods.

`xai-compare` includes a variety of XAI techniques like SHAP, LIME, and Permutation Feature Importance, and introduces advanced comparison techniques such as consistency measurement and feature selection analysis. It is designed to be flexible, easy to integrate, and ideal for enhancing model transparency and interpretability across various applications.

## XAI-Compare Documentation
You can find our ReadTheDocs (RTD) [documentation here](https://xai-compare.readthedocs.io/en/latest/).


## Installation

The package can be installed from [PyPI](https://pypi.org/pypi/xai-compare):

Using pip:
```bash
pip install xai-compare
```

## Explainers

`xai-compare` supports three popular model-agnostic XAI methods:

### SHAP
- SHAP values provide global interpretations of a model's output by attributing each feature's contribution to the predicted outcome.
- Depending on the model type, the script initializes an appropriate explainer such as `shap.TreeExplainer` for tree-based models, `shap.LinearExplainer` for linear models, or `shap.KernelExplainer` for more general models. It then uses SHAP to analyze and explain the behavior of the model.

### LIME
- LIME provides local interpretations of individual predictions by approximating the model's behavior around specific data points.
- The script initializes a LimeTabularExplainer and explains local predictions of the model using LIME.

### Permutation Feature Importance
- Permutation Feature Importance assesses the impact of each feature on a model’s prediction by measuring the decrease in the model’s performance when the values of a feature are randomly shuffled.
- The script measures this dependency by calculating the decrease in model performance after permuting each feature, averaged over multiple permutations.



## Comparison techniques

### Feature selection

The FeatureSelection class in `xai-compare` is a robust tool for optimizing machine learning models by identifying and prioritizing the most influential features. This class leverages a variety of explainers, including SHAP, LIME, and Permutation Importance, to evaluate feature relevance systematically. It facilitates the iterative removal of less significant features, allowing users to understand the impact of each feature on model performance. This approach not only improves model efficiency but also enhances interpretability, making it easier to understand and justify model decisions.


<div align="center">
    <img src="https://github.com/emunaran/xai-compare/raw/main/docs/images/Feature_selection_wf.png" alt="Feature Selection Workflow" width="700"/>
    <p style="color: #808080;">Feature Selection Workflow</p>
</div>


### Consistency
The Consistency class assesses the stability and reliability of explanations provided by various explainers across different splits of data. This class is crucial for determining whether the insights provided by model explainers are consistent regardless of data variances. 

<div align="center">
    <img src="https://github.com/emunaran/xai-compare/raw/main/docs/images/Consistency_wf.png" alt="Consistency Measurement Workflow" width="700"/>
    <p style="color: #808080;">Consistency Measurement Workflow</p>
</div>


## Sample notebooks
The notebooks below demonstrate different use cases for `xai-compare` package. For hands-on experience and to explore the notebooks in detail, visit the notebooks directory in the repository.

[Feature Selection Comparison Notebook](
xai_compare/demo_notebooks/comparison_feature_selection.ipynb)

[Consistency Comparison Notebook](
xai_compare/demo_notebooks/comparison_consistency.ipynb)

[Main Demo Notebook](
xai_compare/demo_notebooks/main_demo.ipynb)


## Call for Contributors
We're seeking individuals with expertise in machine learning, preferably explainable artificial intelligence (XAI), and proficiency in Python programming. If you have a background in these areas and are passionate about enhancing machine learning model transparency, we welcome your contributions. Join us in shaping the future of interpretable AI. 


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments
- SHAP and LIME libraries are used for model interpretability.

