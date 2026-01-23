from time import time
from turtle import fd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import matplotlib.pyplot as plt
from collections import Counter

df = pd.read_csv('creditcard.csv')
#print(df.head())
print(df.shape) #tensor metrics

#1. preprocessing of data

#checking for missing values
df.isnull().sum()
#for any missing values
df.isnull().values.any()
#no missing or null values found

X = df.drop("Class", axis=1)
y = df["Class"]
#print(X)
#print(y)


#train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, #random state for reproducibility - matters for randomly generating data during execution of the code
    stratify=y)#for handling the imbalance. here the imbalalance = 100:0


#normalization
X_train_mean = X_train.mean()
X_train_std = X_train.std()

#formula = x-mean/ std
X_train = (X_train-X_train_mean)/X_train_std
X_test = (X_test - X_train_mean)/X_train_std
X_train=X_train.values
X_test=X_test.values
y_train=y_train.values
y_test=y_test.values

print("Train mean (per feature):")
print(X_train.mean(axis=0))

print("\nTrain std (per feature):")
print(X_train.std(axis=0))

# use a class weighted
counter = Counter(y_train)
n_samples = len(y_train)
class_weights = {
    0: n_samples / (2 * counter[0]),
    1: n_samples / (2 * counter[1])
}
print("Class Weights:", class_weights)


#implementing a perceptron from scratch
class percep:
    def __init__(self, learning_rate=0.001, epochs=50):
        self.lr = learning_rate
        self.epochs = epochs
    
    def fit(self, X, y, class_weights=None):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        for epoch in range(self.epochs):
            for xi, yi in zip(X, y):
                linear_out = np.dot(xi, self.weights)+self.bias
                #y=xi.w+b
                y_pred = 1 if linear_out>=0 else 0
                
                if class_weights:
                    update=self.lr*class_weights[yi]*(yi - y_pred)
                else:
                    update = self.lr*(yi - y_pred)
                #update w&b
                #update = self.lr*(yi-y_pred)
                self.weights+=update *xi
                self.bias+=update
    
    def predict(self, X):
        preds = []
        for x in X:
            linear_out = np.dot(x, self.weights)+self.bias
            preds.append(1 if linear_out>=0 else 0)
        return np.array(preds)

#using stratify
perceptron = percep(learning_rate=0.001, epochs=10)
perceptron.fit(X_train, y_train)

#using class weights
perceptron_weights = percep(learning_rate=0.001, epochs=10)
perceptron_weights.fit(X_train, y_train, class_weights)

#eval metrics
y_train_pred=perceptron.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
print("Train Accuracy of the percep:", train_acc)

y_test_pred=perceptron.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print("Test Accuracy of percep:", test_acc)

percep_metrics = {
    "Accuracy": accuracy_score(y_test, y_test_pred),
    "Precision": precision_score(y_test, y_test_pred),
    "Recall": recall_score(y_test, y_test_pred),
    "F1-score": f1_score(y_test, y_test_pred)
}
print("\nPerceptron Test Metrics:")
for k, v in percep_metrics.items():
    print(f"{k}: {v:.4f}")


#mlp from scratch

##1. activation fn defns



class MLP:
    def __init__(self, X, y, X_val, y_val, hidden_size=16, class_weights=None):
        #constructor defines i/p, o/p, and the hidden layer has 16 neurons
        #bias is trained, not fixed. so the computation of the bias is done while iterations
        self.X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        self.y = y.reshape(-1, 1) #to column vectors

        self.X_val = np.concatenate((X_val, np.ones((X_val.shape[0], 1))), axis=1)
        self.y_val = y_val.reshape(-1, 1)

        self.n_samples = self.X.shape[0]
        self.hidden_size = hidden_size
        self.class_weights = class_weights
        #start off with random weights, later train them to achieve optimal acc
        self.W1 = np.random.randn(self.X.shape[1], hidden_size) * 0.01
        self.W2 = np.random.randn(hidden_size, 1) * 0.01

        #to store train&val losses
        self.train_loss = []
        self.val_loss = []
        
    #activation fns
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoid_prime(self, p):
        return p*(1-p)
    
    #there can be multiple loss fns fro classification and regression separately.
    #here, losss fn =binary cross entropy
    def loss(self, y_hat, y):
        if self.class_weights:
            w0 = self.class_weights[0]
            w1 = self.class_weights[1]
        else:
            w0 = 1
            w1 = 1
        return -np.mean(
            w1*y * np.log(y_hat + 1e-8) +
            w0*(1 - y) * np.log(1 - y_hat + 1e-8) #defn of the loss fn 
        )  
        
    #forward pass
    def forward(self, X):
        self.z1 = X @ self.W1 #z1 = sigma(x, w1)
        self.a1 = self.sigmoid(self.z1) #a1=activ_fn(z1)

        self.z2 = self.a1 @ self.W2
        self.y_hat = self.sigmoid(self.z2)#y_pred 

        return self.y_hat

    #backprop
    def backward(self, X, y, lr):
        if self.class_weights is not None:
            sample_weights = (
                y * self.class_weights[1] +
                (1 - y) * self.class_weights[0]
            )
            dz2 = (self.y_hat - y) * sample_weights
        else:
            dz2 = self.y_hat - y

        dW2 = self.a1.T @ dz2 / len(X)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.sigmoid_prime(self.a1)
        dW1 = X.T @ dz1 / len(X)

        self.W2 -= lr * dW2
        self.W1 -= lr * dW1


    def train(self, epochs=30, lr=0.01, batch_size=64):
        #batch size defines after how many samples must the gd be applied and updation happen
        
        for epoch in range(epochs):
            start = time.time()

            indices = np.random.permutation(self.n_samples) #shuffle the data
            X_shuffled = self.X[indices]
            y_shuffled = self.y[indices]

            for i in range(0, self.n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                #grad 1 applied 
                self.forward(X_batch)
                self.backward(X_batch, y_batch, lr)

            #loss calc
            train_loss = self.loss(self.forward(self.X), self.y)
            val_loss = self.loss(self.forward(self.X_val), self.y_val)

            self.train_loss.append(train_loss)
            self.val_loss.append(val_loss)

            print(
                f"Epoch {epoch+1}: "
                f"Train Loss = {train_loss:.4f}, "
                f"Val Loss = {val_loss:.4f}, "
                f"Time = {round(time.time()-start,2)}s"
            )
    def predict(self, X):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        y_hat = self.forward(X)
        return (y_hat >= 0.5).astype(int) #prob is converted to class
    
    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y.reshape(-1, 1))

mlp = MLP(X_train, y_train, X_test, y_test)
mlp.train(epochs=50, lr=0.01, batch_size=64)
mlp_weighted = MLP(X_train, y_train, X_test, y_test, class_weights=class_weights)
mlp_weighted.train(epochs=50, lr=0.01, batch_size=64)

y_pred_mlp = mlp.predict(X_test)




#eval metrics mlp
y_test_pred_mlp = mlp.predict(X_test)

mlp_metrics = {
    "Accuracy": accuracy_score(y_test, y_test_pred_mlp),
    "Precision": precision_score(y_test, y_test_pred_mlp, zero_division=0),
    "Recall": recall_score(y_test, y_test_pred_mlp),
    "F1-score": f1_score(y_test, y_test_pred_mlp)
}
labels = list(percep_metrics.keys())

percep_values = list(percep_metrics.values())
mlp_values = list(mlp_metrics.values())

print("\nMLP Test Metrics:")
for k, v in mlp_metrics.items():
    print(f"{k}: {v:.4f}")


#comparison of stratify vs class weights
def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    print(f"\n{name}")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1-score :", f1_score(y_test, y_pred))

evaluate(mlp, X_test, y_test, "MLP: Stratified ")
evaluate(mlp_weighted, X_test, y_test, "MLP Stratified + Class Weight")


#visualtizations

#train vs validation loss plot

plt.figure()
plt.plot(mlp.train_loss, label="Training Loss")
plt.plot(mlp.val_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("MLP Loss Convergence")
plt.legend()
plt.show()
plt.savefig("mlp_loss_convergence.png")

#train vs validation loss plot of class weights
plt.figure()
plt.plot(mlp_weighted.train_loss, label="Training Loss")
plt.plot(mlp_weighted.val_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("MLP(class weighted) Loss Convergence")
plt.legend()
plt.show()
plt.savefig("mlpwt_loss_convergence.png")


#stratify vs stratify + class weights comparison

plt.figure(figsize=(9, 5))
# Stratified only
plt.plot(mlp.train_loss, label="strat_train", linestyle="--")
plt.plot(mlp.val_loss, label="strat_loss", linestyle="--")
# Stratified + class weights
plt.plot(mlp_weighted.train_loss, label="classwt_train")
plt.plot(mlp_weighted.val_loss, label="classwt_val")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Convergence of MLP: statify vs class weights")
plt.legend()
plt.grid(alpha=0.4)
plt.savefig("stratVSclasswt.png")
plt.show()


#accuracy, precision, recall, f1-score comparison for percp vs mlp

metrics_labels = ["Accuracy", "Precision", "Recall", "F1-score"]
percep_values = [percep_metrics[m] for m in metrics_labels]
mlp_values = [mlp_metrics[m] for m in metrics_labels]
x = np.arange(len(metrics_labels))
width = 0.35
plt.figure(figsize=(8, 5))
plt.bar(x - width/2, percep_values, width, label="Perceptron")
plt.bar(x + width/2, mlp_values, width, label="MLP")
plt.xticks(x, metrics_labels)
plt.ylabel("Score")
plt.title("Performance Comparison: Perceptron vs MLP")
plt.legend()
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()
plt.savefig("performance_comparison.png")

    


