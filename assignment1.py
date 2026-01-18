from time import time
from turtle import fd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
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

print(X_train)



#implementing a perceptron from scratch
class percep:
    def __init__(self, learning_rate=0.001, epochs=50):
        self.lr = learning_rate
        self.epochs = epochs
    
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        for epoch in range(self.epochs):
            for xi, yi in zip(X, y):
                linear_out = np.dot(xi, self.weights)+self.bias
                #y=xi.w+b
                y_pred = 1 if linear_out>=0 else 0
                
                #update w&b
                update = self.lr*(yi-y_pred)
                self.weights+=update *xi
                self.bias+=update
    
    def predict(self, X):
        preds = []
        for x in X:
            linear_out = np.dot(x, self.weights)+self.bias
            preds.append(1 if linear_out>=0 else 0)
        return np.array(preds)
    
perceptron = percep(learning_rate=0.001, epochs=10)
perceptron.fit(X_train, y_train)

#eval metrics
y_train_pred=perceptron.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
print("Train Accuracy of the percep:", train_acc)

y_test_pred=perceptron.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print("Test Accuracy of percep:", test_acc)
                

#mlp from scratch

##1. activation fn defns



class MLP:
    def __init__(self, X, y, X_val, y_val, hidden_size=16):
        # Add bias term
        self.X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        self.y = y.reshape(-1, 1)

        self.X_val = np.concatenate((X_val, np.ones((X_val.shape[0], 1))), axis=1)
        self.y_val = y_val.reshape(-1, 1)

        self.n_samples = self.X.shape[0]
        self.hidden_size = hidden_size

        # Initialize weights
        self.W1 = np.random.randn(self.X.shape[1], hidden_size) * 0.01
        self.W2 = np.random.randn(hidden_size, 1) * 0.01

        # Metrics
        self.train_loss = []
        self.val_loss = []
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoid_prime(self, p):
        return p*(1-p)
    
    def loss(self, y_hat, y):
        return -np.mean(
            y * np.log(y_hat + 1e-8) +
            (1 - y) * np.log(1 - y_hat + 1e-8)
        )  
    def forward(self, X):
        self.z1 = X @ self.W1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = self.a1 @ self.W2
        self.y_hat = self.sigmoid(self.z2)

        return self.y_hat

    def backward(self, X, y, lr):
        # Output layer gradient
        dz2 = self.y_hat - y
        dW2 = self.a1.T @ dz2 / len(X)

        # Hidden layer gradient
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.sigmoid_prime(self.a1)
        dW1 = X.T @ dz1 / len(X)

        # Gradient descent update
        self.W2 -= lr * dW2
        self.W1 -= lr * dW1

    def train(self, epochs=30, lr=0.01, batch_size=64):
        for epoch in range(epochs):
            start = time.time()

            indices = np.random.permutation(self.n_samples)
            X_shuffled = self.X[indices]
            y_shuffled = self.y[indices]

            for i in range(0, self.n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                self.forward(X_batch)
                self.backward(X_batch, y_batch, lr)

            # Track loss
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
        return (y_hat >= 0.5).astype(int)
    
    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y.reshape(-1, 1))

mlp = MLP(X_train, y_train, X_test, y_test)
mlp.train(epochs=30, lr=0.01, batch_size=64)

print("MLP Training Accuracy:", mlp.accuracy(X_train, y_train))
print("MLP Test Accuracy:", mlp.accuracy(X_test, y_test))

    
    

#gradient descent