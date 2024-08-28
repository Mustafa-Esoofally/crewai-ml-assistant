```
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
data = pd.read_csv('customer_churn_data.csv')

# Preprocess data
X = data.drop(['churn'], axis=1)
y = data['churn']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[('onehot', OneHotEncoder(handle_unknown='ignore'), ['feature1', 'feature2', ...])], remainder='passthrough')

# Define models
models = [
    ('logistic_regression', Pipeline([('preprocessor', preprocessor), ('logistic_regression', LogisticRegression())])),
    ('decision_tree', Pipeline([('preprocessor', preprocessor), ('decision_tree', DecisionTreeClassifier())])),
    ('random_forest', Pipeline([('preprocessor', preprocessor), ('random_forest', RandomForestClassifier())])),
    ('gradient_boosting', Pipeline([('preprocessor', preprocessor), ('gradient_boosting', GradientBoostingClassifier())])),
    ('neural_network', Pipeline([('preprocessor', preprocessor), ('neural_network', MLPClassifier())])),
    ('support_vector_machine', Pipeline([('preprocessor', preprocessor), ('support_vector_machine', SVC())]))
]

# Train and evaluate models
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f'Model: {name}')
    print(f'Accuracy: {accuracy_score(y_val, y_pred):.3f}')
    print(f'Classification Report:\n{classification_report(y_val, y_pred)}')
    print(f'Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}')
    print()
```

This code provides a starter template for customer churn prediction using the recommended machine learning models. It includes data loading, preprocessing, model definition, and a basic training loop. The code can be customized and extended to suit the specific needs of the project.