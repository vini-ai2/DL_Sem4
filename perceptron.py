from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

wine = datasets.load_wine()

X = wine.data
Y = wine.target

print(wine.DESCR)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)

y_pred = perceptron.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

report = classification_report(Y_test, y_pred, target_names = wine.target_names)
print("Classification Report:")
print(report)

#a great variety is always preferred to a large volume 
