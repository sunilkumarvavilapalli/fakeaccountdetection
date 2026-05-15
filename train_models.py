import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import joblib

# Load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Define target
target = 'fake'

# Fill NaN values
train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

# Feature engineering
for df in [train_df, test_df]:
    df['#followers'] += 1
    df['#follows'] += 1
    df['#posts'] += 1

    df['following_follower_ratio'] = df['#follows'] / df['#followers']
    df['posts_per_follower'] = df['#posts'] / df['#followers']
    df['activity_score'] = df['#posts'] / df['#follows']
    df['engagement_ratio'] = df['#followers'] / df['#posts']

# Define features
FEATURES = [
    '#followers', '#follows', '#posts',
    'following_follower_ratio',
    'posts_per_follower',
    'activity_score',
    'engagement_ratio'
]

# Prepare data
X = train_df[FEATURES]
y = train_df[target]

X_test_final = test_df[FEATURES]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test_final)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train models
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train, y_train)

print("Training Logistic Regression...")
lr = LogisticRegression(
    max_iter=2000,
    class_weight='balanced'
)
lr.fit(X_train, y_train)

print("Training Naive Bayes...")
nb = GaussianNB()
nb.fit(X_train, y_train)

print("Training Ensemble Model...")
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('lr', lr),
        ('nb', nb)
    ],
    voting='soft',
    weights=[0.6, 0.3, 0.1]
)
ensemble.fit(X_train, y_train)

# Save models
print("Saving models...")
joblib.dump(ensemble, "fake_account_rf_lr_nb.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(FEATURES, "feature_columns.pkl")

print("Models saved successfully!")
