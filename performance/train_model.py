import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def train():
    # 1. Load the dataset
    df = pd.read_csv('Student_Performance.csv')

    # 2. Convert 'Extracurricular Activities' to numeric (Yes=1, No=0)
    df['Extracurricular Activities'] = LabelEncoder().fit_transform(df['Extracurricular Activities'])

    # 3. Define features (X) and target (y)
    X = df[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
    y = df['Performance Index']

    # 4. Split data into train/test (optional, here mostly for completeness)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Train the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # 6. Save the model
    joblib.dump(model, 'performance/model.pkl')

    print("Model trained and saved successfully!")

# Run this script from Django shell or directly
if __name__ == "__main__":
    train()
