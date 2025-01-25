import numpy as np
import pandas as pd
import sys

try:
    import matplotlib.pyplot as plt
except ImportError:
    sys.stderr.write("Matplotlib is not installed. Please install it using `pip install matplotlib`.\n")
    sys.exit(1)

try:
    import seaborn as sns
except ImportError:
    sys.stderr.write("Seaborn is not installed. Please install it using `pip install seaborn`.\n")
    sys.exit(1)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import pickle  # For model saving
import streamlit as st  # For prototype UI

# Phase 1: Data Loading and Preprocessing
def load_and_preprocess_data(filepath):
    """Load dataset and perform preprocessing."""
    data = pd.read_csv(filepath)
    data.dropna(inplace=True)  # Handle missing values
    return data

# Phase 2: Feature Engineering
def feature_engineering(data):
    """Perform feature scaling and additional transformations."""
    X = data.drop('target', axis=1)  # Replace 'target' with your target variable column
    y = data['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# Phase 3: Data Analysis and Visualization
def visualize_data(data):
    """Generate visualizations to understand the dataset."""
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Phase 4: Model Building
def train_models(X_train, y_train):
    """Train multiple models and return them with their performances."""
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier()
    }
    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = cross_val_score(model, X_train, y_train, cv=5)
        scores[name] = score.mean()
    return models, scores

# Phase 5: Evaluation and Visualization
def evaluate_model(model, X_test, y_test):
    """Evaluate model and visualize results."""
    y_pred = model.predict(X_test)
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    st.write("### ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    st.pyplot(fig)

# Streamlit UI
def main():
    st.title("Supervised Learning Model")

    filepath = st.text_input("Enter the path to your dataset (e.g., data.csv):")

    if filepath:
        try:
            data = load_and_preprocess_data(filepath)

            st.write("### Dataset Preview")
            st.write(data.head())

            st.write("### Data Visualization")
            visualize_data(data)

            X, y = feature_engineering(data)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            models, scores = train_models(X_train, y_train)
            best_model_name = max(scores, key=scores.get)
            best_model = models[best_model_name]

            st.write(f"### Best Model: {best_model_name} with score {scores[best_model_name]:.4f}")

            evaluate_model(best_model, X_test, y_test)

            # Save the model
            with open('model.pkl', 'wb') as f:
                pickle.dump(best_model, f)

            st.write("### Model Saved as model.pkl")

            st.write("### Make Predictions")
            user_input = st.text_input("Enter features as comma-separated values:")

            if user_input:
                features = [float(x) for x in user_input.split(',')]
                prediction = best_model.predict([features])[0]
                st.write(f"Prediction: {prediction}")
        except FileNotFoundError:
            st.error("Error: The file path provided is incorrect or the file does not exist.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
