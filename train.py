import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import joblib
import matplotlib.pyplot as plt

os.makedirs("Model", exist_ok=True)
os.makedirs("Results", exist_ok=True)

drug_df = pd.read_csv("Data/drug.csv")
drug_df = drug_df.sample(frac=1)

X = drug_df.drop("Drug", axis=1).values
y = drug_df.Drug.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=125)

cat_col = [1,2,3]
num_col = [0,4]
transform = ColumnTransformer([
    ("encoder", OrdinalEncoder(), cat_col),
    ("num_imputer", SimpleImputer(strategy="median"), num_col),
    ("num_scaler", StandardScaler(), num_col),
])
pipe = Pipeline([
    ("preprocessing", transform),
    ("model", RandomForestClassifier(n_estimators=100, random_state=125)),
])
pipe.fit(X_train, y_train)

predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"\nAccuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}.")

cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.savefig("Results/model_results.png", dpi=120)

joblib.dump(pipe, "Model/drug_pipeline.joblib")
print("Training complete — model saved to Model/drug_pipeline.joblib")

try:
    import skops.io as sio
    sio.dump(pipe, "Model/drug_pipeline.skops")
except ImportError:
    print("skops not installed, skipping skops model save.")