import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('Iris.csv')
print(df.columns.tolist())
# Drop the Id column if it exists
df = df.drop(['Id'], axis=1, errors='ignore')

# Features and target
X = df.drop('Species', axis=1)
y = df['Species']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)

# Save model
joblib.dump(model, 'iris_model.pkl')
print("Model saved as iris_model.pkl")
