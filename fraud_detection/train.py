import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

df = pd.read_csv(r'C:\Users\user\OneDrive\Desktop\PayGuard\data\creditcard.csv')
print(df.head())
print (df.info())
print (df.columns)
print("Dataset shape:", df.shape)
print("Number of missing values:\n", df.isnull().sum().sum())
print("Fraude distribution:\n", df['Class'].value_counts())
fraud_percentage = df["Class"].mean() * 100
print(f"Percentage of fraudulent transactions: {fraud_percentage:.3f}%")

df = df.sort_values("Time").reset_index(drop=True)
X=df.drop(columns=["Class"])
Y=df["Class"]

Train_end = int(len(df)*0.70)
val_end= int(len(df)*0.85)

X_train = X.iloc[:Train_end]
X_val = X.iloc[Train_end:val_end]
X_test = X.iloc[val_end:]

Y_train = Y.iloc[:Train_end]
Y_val = Y.iloc[Train_end:val_end]
Y_test = Y.iloc[val_end:]

for name , target in[
    ("Train", Y_train),
    ("Validation", Y_val),
    ("Test", Y_test)
]:
    print(f"{name} set")
    print("Total Transactions:", len(target))
    print("Fraud Transactions:", target.sum())
    print(f"Fraud Percentage: {target.mean()*100:.3f}%")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(X_train, Y_train)
y_pred = dummy_model.predict(X_val)
y_score = dummy_model.predict_proba(X_val)[:, 1]

print("\nDummy classifier result")
print("Accuracy:", accuracy_score(Y_val, y_pred))
print("Precision:", precision_score(Y_val, y_pred, zero_division=0))
print("Recall:", recall_score(Y_val, y_pred, zero_division=0))
print("F1-Score:", f1_score(Y_val, y_pred, zero_division=0))
print("Average Precision:", average_precision_score(Y_val, y_score))
cm_dummy = confusion_matrix(Y_val, y_pred, labels=[0, 1])

print("\nDummy Classifier Confusion Matrix:")
print(cm_dummy)

tn, fp, fn, tp = cm_dummy.ravel()

print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

#logistic regression model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

logistic_model = make_pipeline(
    StandardScaler(), 
    LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
)
logistic_model.fit(X_train, Y_train)
y_pred_lr = logistic_model.predict(X_val)
y_score_lr = logistic_model.predict_proba(X_val)[:, 1]

print("\nLogistic Regression result")
print("Accuracy:", accuracy_score(Y_val, y_pred_lr))
print("Precision:", precision_score(Y_val, y_pred_lr))
print("Recall:", recall_score(Y_val, y_pred_lr))
print("F1-Score:", f1_score(Y_val, y_pred_lr))
print("Average Precision:", average_precision_score(Y_val, y_score_lr))

cm = confusion_matrix(Y_val, y_pred_lr)
print("\nConfusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")



thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]


for threshold in thresholds:

    
    predictions = (y_score_lr >= threshold).astype(int)

  
    precision = precision_score(
        Y_val, predictions, zero_division=0
    )

    recall = recall_score(
        Y_val, predictions, zero_division=0
    )

    f1 = f1_score(
        Y_val, predictions, zero_division=0
    )

  
    tn, fp, fn, tp = confusion_matrix(
        Y_val, predictions
    ).ravel()

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")


best_threshold = 0.95


y_score_test = logistic_model.predict_proba(X_test)[:, 1]
y_pred_test = (y_score_test >= best_threshold).astype(int)

print("\nFINAL TEST RESULTS: LOGISTIC REGRESSION")
print("Selected Threshold:", best_threshold)

print("Accuracy:", accuracy_score(Y_test, y_pred_test))
print("Precision:", precision_score(Y_test, y_pred_test, zero_division=0))
print("Recall:", recall_score(Y_test, y_pred_test, zero_division=0))
print("F1-Score:", f1_score(Y_test, y_pred_test, zero_division=0))
print("Average Precision:", average_precision_score(Y_test, y_score_test))

cm_test = confusion_matrix(Y_test, y_pred_test, labels=[0, 1])

print("\nTest Confusion Matrix:")
print(cm_test)

tn, fp, fn, tp = cm_test.ravel()
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

# Save final model
import joblib
from pathlib import Path

model_dir = Path(__file__).resolve().parent / "models"
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "fraud_model.joblib"

joblib.dump(
    {
        "model": logistic_model,
        "threshold": best_threshold,
        "feature_columns": X_train.columns.tolist()
    },
    model_path
)

print("\nModel saved successfully!")
print("Model location:", model_path)
