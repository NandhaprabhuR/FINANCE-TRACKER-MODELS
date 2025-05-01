import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

def preprocess_text(text):
    # Convert to lowercase and remove extra spaces
    text = text.lower().strip()
    # Optionally remove the amount part for training (e.g., split at "$" and take the first part)
    if '$' in text:
        text = text.split('$')[0].strip()
    return text

try:
    # Load dataset
    df = pd.read_csv(r"C:\Users\G N Kavin\Desktop\financeexpense\personal_expense_classification.csv")

    # Check for missing values
    print("Missing values:\n", df.isnull().sum())

    # Drop rows with missing values (if any)
    df = df.dropna()

    # Preprocess descriptions
    df['description'] = df['description'].apply(preprocess_text)

    # Split data
    X = df['description']
    y = df['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create pipeline: TF-IDF + OneVsRestClassifier with Logistic Regression
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear')))
    ])

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    # Save model
    joblib.dump(model, 'expense_classifier.pkl')
    print("Model saved as expense_classifier.pkl")

except FileNotFoundError as e:
    print(f"Error: Dataset file not found. Please check the file path: {e}")
except Exception as e:
    print(f"An error occurred: {e}")