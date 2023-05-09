# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from Classifier.RandomForest import RandomForest
from ParameterAnalyzer.SHAPTree import SHAPTree

# Load the Breast Cancer Wisconsin dataset
data = load_breast_cancer()
X, y = pd.DataFrame(data.data, columns=data.feature_names), pd.Series(data.target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier on the training set
model = RandomForest().get_classifier()
model.fit(X_train, y_train)

shap_tree_analyzer = SHAPTree(model)
shap_tree_analyzer.plot_explainer(X_test)