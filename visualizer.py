import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeClassifier
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="ML Visualizer", layout="wide")
plt.style.use("seaborn-v0_8")

param_config = {
    "Logistic Regression": [
        {"type": "slider", "label": "C", "key": "C", "min": 0.001, "max": 100, "default": 1},
        # FIX: removed solvers incompatible with l1/elasticnet; lbfgs is safest default
        {"type": "selectbox", "label": "Solver", "key": "solver", "options": ['lbfgs', 'saga']},
        # FIX: l1 and elasticnet only work with saga; lbfgs only supports l2/None
        {"type": "selectbox", "label": "Penalty", "key": "penalty", "options": ["l2", "l1", "elasticnet"]},
        {"type": "slider", "label": "max_iter", "key": "max_iter", "min": 100, "max": 2000, "default": 500},
        {"type": "selectbox", "label": "class_weight", "key": "class_weight", "options": ["balanced"]},
        # FIX: removed multi_class — deprecated and removed in sklearn >= 1.5
    ],
    "SVM": [
        {"type": "slider", "label": "C", "key": "C", "min": 0.001, "max": 1000, "default": 1},
        # FIX: removed 'precomputed' — requires a kernel matrix input, not raw data
        {"type": "selectbox", "label": "kernel", "key": "kernel", "options": ['linear', 'rbf', 'poly', 'sigmoid']},
        {"type": "slider", "label": "max_iter", "key": "max_iter", "min": -1, "max": 10000, "default": -1},
    ],
    "Decision Tree": [
        {"type": "slider", "label": "max_depth", "key": "max_depth", "min": 1, "max": 1000, "default": 200},
        {"type": "slider", "label": "min_sample_split", "key": "min_samples_split", "min": 2, "max": 50, "default": 2},
        {"type": "slider", "label": "min_samples_leaf", "key": "min_samples_leaf", "min": 1, "max": 50, "default": 2},
        {"type": "selectbox", "label": "criterion", "key": "criterion", "options": ['gini', 'entropy', 'log_loss']},
        # FIX: removed 'auto' — removed in sklearn 1.2
        {"type": "selectbox", "label": "max_features", "key": "max_features", "options": ['sqrt', 'log2']},
    ],
    "Random Forest": [
        {"type": "slider", "label": "n_estimators", "key": "n_estimators", "min": 10, "max": 1000, "default": 200},
        {"type": "slider", "label": "max_depth", "key": "max_depth", "min": 1, "max": 1000, "default": 50},
        # FIX: key was "max_sample_split" — corrected to "min_samples_split"
        {"type": "slider", "label": "min_samples_split", "key": "min_samples_split", "min": 2, "max": 50, "default": 2},
        {"type": "slider", "label": "min_samples_leaf", "key": "min_samples_leaf", "min": 1, "max": 50, "default": 2},
        {"type": "selectbox", "label": "criterion", "key": "criterion", "options": ['gini', 'entropy', 'log_loss']},
        # FIX: removed 'auto' — removed in sklearn 1.2
        {"type": "selectbox", "label": "max_features", "key": "max_features", "options": ['sqrt', 'log2']},
    ],
    "KNN": [
        {"type": "slider", "label": "n_neighbors", "key": "n_neighbors", "min": 1, "max": 1000, "default": 5},
        {"type": "selectbox", "label": "weights", "key": "weights", "options": ['uniform', 'distance']},
        # FIX: typo 'manhatton' → 'manhattan'
        {"type": "selectbox", "label": "metric", "key": "metric", "options": ['euclidean', 'manhattan']},
    ],
    "Linear Regression": [
        {"type": "selectbox", "label": "fit_intercept", "key": "fit_intercept", "options": [True, False]},
        {"type": "selectbox", "label": "copy_X", "key": "copy_X", "options": [True, False]},
        {"type": "slider", "label": "n_jobs", "key": "n_jobs", "min": -1, "max": 8, "default": -1},
        {"type": "selectbox", "label": "positive", "key": "positive", "options": [True, False]},
    ],
    "Ridge Regression": [
        {"type": "slider", "label": "alpha", "key": "alpha", "min": 0.0001, "max": 1000.0, "default": 1.0},
        {"type": "selectbox", "label": "solver", "key": "solver", "options": ["auto", "svd", "lsqr", "cholesky", "sag", "saga"]},
        {"type": "slider", "label": "max_iter", "key": "max_iter", "min": 100, "max": 5000, "default": 500},
        {"type": "slider", "label": "tol", "key": "tol", "min": 1e-5, "max": 1e-1, "default": 1e-3},
    ],
}


def load_dataset(name):
    if name == "binary":
        x, y = make_blobs(n_features=2, centers=2, random_state=2)
        return x, y, ['Feature 1', 'Feature 2']
    if name == 'multiclass':
        x, y = make_blobs(n_features=2, centers=3, random_state=2)
        return x, y, ['Feature 1', 'Feature 2']
    if name == "Iris":
        data = load_iris()
        X = data.data[:, :2]
        y = data.target
        return X, y, ["Sepal Length", "Sepal Width"]


def meshgrid(x):
    a = np.arange(x[:, 0].min() - 1, x[:, 0].max() + 1, 0.02)
    b = np.arange(x[:, 1].min() - 1, x[:, 1].max() + 1, 0.02)
    xx, yy = np.meshgrid(a, b)
    pts = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, pts


def get_model(name, **params):
    if name == "Logistic Regression":
        # FIX: enforce saga for l1/elasticnet; lbfgs for l2
        penalty = params.get("penalty", "l2")
        if penalty in ("l1", "elasticnet"):
            params["solver"] = "saga"
        else:
            params["solver"] = "lbfgs"
        return LogisticRegression(**params)
    if name == "SVM":
        return SVC(**params)
    if name == "Decision Tree":
        return DecisionTreeClassifier(**params)
    if name == "KNN":
        return KNeighborsClassifier(**params)
    if name == "Random Forest":
        return RandomForestClassifier(**params)
    if name == "Linear Regression":
        return LinearRegression(**params)
    if name == "Ridge Regression":
        return RidgeClassifier(**params)
    return None


st.sidebar.title("ML Classifier Visualizer")

dataset = st.sidebar.selectbox("Dataset", ["binary", "multiclass", "Iris"])
model_name = st.sidebar.selectbox(
    "Model",
    ['Logistic Regression', "SVM", "Decision Tree", "Random Forest", "KNN", "Linear Regression", "Ridge Regression"],
)

selected_params = {}
for param in param_config[model_name]:
    if param["type"] == "slider":
        min_val = param["min"]
        max_val = param["max"]
        default = param["default"]
        if any(isinstance(v, float) for v in [min_val, max_val, default]):
            min_val = float(min_val)
            max_val = float(max_val)
            default = float(default)
        selected_params[param["key"]] = st.sidebar.slider(param["label"], min_val, max_val, default)
    elif param["type"] == "selectbox":
        selected_params[param["key"]] = st.sidebar.selectbox(param["label"], param["options"])

x, y, feat = load_dataset(dataset)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, random_state=42)

model = get_model(model_name, **selected_params)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

col1, col2 = st.columns([1.2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x_scaled[:, 0], x_scaled[:, 1], c=y, cmap="winter", edgecolor="black")
    xx, yy, pts = meshgrid(x_scaled)
    z = model.predict(pts).reshape(xx.shape)
    ax.contour(xx, yy, z, alpha=0.35, cmap='rainbow')
    ax.set_xlabel(feat[0])
    ax.set_ylabel(feat[1])
    st.pyplot(fig)

with col2:
    st.subheader("Performance Metrics")
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.metric("Accuracy", f"{acc:.2%}")
    st.divider()

    m1, m2, m3 = st.columns(3)
    m1.metric("Precision", f"{report['weighted avg']['precision']:.2%}")
    m2.metric("Recall", f"{report['weighted avg']['recall']:.2%}")
    m3.metric("F1 Score", f"{report['weighted avg']['f1-score']:.2%}")
    st.divider()

    with st.expander("Full Classification Report"):
        st.code(classification_report(y_test, y_pred), language="text")
