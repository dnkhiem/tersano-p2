import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def read_data(filename = 'customer_churn_dataset-testing-master.csv'):
    df = pd.read_csv(filename)
    df = df.dropna()
    return df

df = read_data(filename='customer_churn_dataset-testing-master.csv')
# df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
# df['Subscription Type'] = df['Subscription Type'].map({'Basic': 0, 'Standard': 1, 'Premium': 2})
# df['Contract Length'] = df['Contract Length'].map({'Monthly': 0, 'Quarterly': 1, 'Annual': 2})

# Encoding categorical variables

def encode_categorical(df):
    label_encoders = {}
    # Encode other categorical variables using LabelEncoder
    categorical_columns = ['Gender', 'Subscription Type', 'Contract Length', 'Churn']
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
        print(f"Unique values in '{column}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

label_encoders = encode_categorical(df)


# Scaling numerical variables
scaler = StandardScaler()
numerical_columns = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Step 2: Exploratory Data Analysis (EDA)
# Summary statistics
def describe_data(df):
    print(df.describe())

    # Visualizations
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Churn', data=df)
    plt.title('Distribution of Churn')
    plt.show()

    plt.figure(figsize=(14, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()

describe_data(df)

# Step 3: Model Training
# Split the data into training and testing sets
X = df.drop(['CustomerID', 'Churn'], axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# Train Decision Tree model
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred_decision_tree = decision_tree.predict(X_test)

# Train Random Forest model
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)
y_pred_random_forest = random_forest.predict(X_test)

# Step 4: Model Selection and Evaluation
def evaluate_model(y_test, y_pred, model_name):
    print(f"---{model_name}---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")
    print(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report:\n {classification_report(y_test, y_pred)}")
    print("\n")

evaluate_model(y_test, y_pred_log_reg, "Logistic Regression")
evaluate_model(y_test, y_pred_decision_tree, "Decision Tree")
evaluate_model(y_test, y_pred_random_forest, "Random Forest")

# Step 5:  Based on evaluation, we select the Random Forest performed best
"""
The Logistic Regression results is not good.
The Random Forest model performs slightly better than the Decision Tree model across all metrics. 
It has a higher accuracy, precision, recall, and F1 score. 
Additionally, the confusion matrix shows fewer misclassifications with the Random Forest model.

Given these observations, the Random Forest model is the better-performing model and should be selected for predicting customer churn."""


# Step 6: Prediction Function
def predict_churn(new_data):
    new_data_transformed = new_data.drop(['CustomerID'], axis=1)
    # new_data_transformed['Gender'] = new_data_transformed['Gender'].map({'Male': 0, 'Female': 1})
    for column in ['Gender','Subscription Type', 'Contract Length']:
        le = label_encoders[column]
        new_data_transformed[column] = le.transform(new_data_transformed[column])
    
    new_data_transformed[numerical_columns] = scaler.transform(new_data_transformed[numerical_columns])
    prediction = random_forest.predict(new_data_transformed)
    return "Churn" if prediction[0] == 1 else "No Churn"

# Example usage of prediction function
new_customer = pd.DataFrame({
    'CustomerID': [99999],
    'Age': [30],
    'Gender': ['Female'],
    'Tenure': [5],
    'Usage Frequency': [3],
    'Support Calls': [1],
    'Payment Delay': [0],
    'Subscription Type': ['Basic'],
    'Contract Length': ['Monthly'],
    'Total Spend': [200],
    'Last Interaction': [10]
})

print(predict_churn(new_customer))