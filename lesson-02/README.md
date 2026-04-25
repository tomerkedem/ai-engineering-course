# Machine Learning Lab

Simple, runnable demos for NumPy, pandas, and scikit-learn.

This repository contains the practical part of an AI Engineering course.

The goal is not just to run scripts,  
but to understand how Machine Learning systems work in practice.

---

## Core Idea

All examples follow the same structure:

Data → Prepare → Train → Predict → Evaluate

Understanding this flow is more important than any specific algorithm.

---

## Installation

From the repository root, create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```powerShell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```
```bash
python -m venv .venv
```

```bat
REM Windows Command Prompt
.venv\Scripts\activate.bat
```

```bash
# macOS / Linux
source .venv/bin/activate
```
Install dependencies:


```bash
pip install -r requirements.txt
```

Running the programs

Run any script from the project root:


```bash
# Compare NumPy vs Python list performance
python src/1_numpy_performance_demo.py

# Work with pandas DataFrames
python src/2_pandas_dataframe_demo.py

# Linear Regression (continuous prediction)
python src/3_linear_regression_classifier.py

# Logistic Regression (classification)
python src/4_logistic_regression_classifier.py

# Decision Tree
python src/5_decision_tree_classifier.py

# Random Forest
python src/6_random_forest_classifier.py

# K-Means clustering
python src/7_kmeans_clustering.py
```
Some scripts open matplotlib windows or save figures under images/.

Learning Approach

Read → Run → Change → Observe

Do not just read the code.
Modify it and see what happens.