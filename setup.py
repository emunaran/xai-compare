from setuptools import setup, find_packages

setup(
    name='xai-compare',  # This will be the short name used for pip install
    version='0.1.2',
    author='Ran Emuna',
    author_email='emuna.ran@gmail.com',
    description='This repository aims to provide tools for comparing different explainability methods, enhancing the '
                'interpretation of machine learning models.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/emunaran/xai-compare.git',
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    install_requires=[
        'jupyter==1.0.0',
        'lime==0.2.0.1',
        'matplotlib==3.9.1',
        'notebook==7.2.1',
        'numpy==1.26.4',
        'pandas==1.5.3',
        'scikit-learn==1.5.1',
        'scipy==1.14.0',
        'seaborn==0.13.2',
        'shap==0.44.0',
        'interpret==0.5.0',
        'interpret-community==0.31.0',
        'interpret-core==0.5.0',

        'sphinx==7.4.7',
        'livereload==2.7.0',
        'sphinx-rtd-theme==2.0.0',
        'numpydoc==1.7.0',
        'nbsphinx==0.9.4',
        'myst-parser==3.0.1',
        'sphinx-github-changelog==1.3.0'
    ],
    package_data={
        'xai_compare': [
            'data/diabetes/diabetes.csv',
            'data/fico/train.csv',
            'data/fico/test.csv',
            'data/german_credit_score/german_credit.csv'
        ],
    },
)
