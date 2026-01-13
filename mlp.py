#1. incorporate visualizations on training data
#2. accordingly, make some pre processing decisions based on that
#3. ensure that the class distribution is retained in the train test splits --- stratify=y
#4. include train, validation, accuracy and loss plots

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

wine = datasets.load_wine()

X = wine.data
Y = wine.target

print(wine.DESCR)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

mlp = MLPClassifier( hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)

mlp.fit(X_train, Y_train)

y_pred = mlp.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

report = classification_report(Y_test, y_pred, target_names = wine.target_names)
print("Classification Report:")
print(report)

#viz 1 - class distr 
unique, counts = np.unique(Y, return_counts=True)

plt.bar(unique, counts)
plt.xticks(unique, wine.target_names)
plt.title("Class Distribution in Wine Dataset")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()


#viz 2 - confusion matrix
cm = confusion_matrix(Y_test, y_pred)
df_cm = pd.DataFrame(cm, index=wine.target_names, columns=wine.target_names)
plt.figure(figsize=(8,6))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


#viz 3 - loss curve
plt.plot(mlp.loss_curve_)
plt.title("MLP Classifier Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

#viz 4-train vs validation accuracy
plt.plot(mlp.validation_scores_)
plt.title("Validation Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.show()


#viz 5 - correlation heatmap
import seaborn as sns
df = pd.DataFrame(X, columns=wine.feature_names)
corr = df.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()




