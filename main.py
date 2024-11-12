import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data and create DataFrame
data = pd.read_csv('Housing.csv')
df = pd.DataFrame(data[['price', 'area', 'bedrooms', 'bathrooms']])

# Normalize area and price
df["area"] = (df["area"] - df["area"].min()) / (df["area"].max() - df["area"].min())
y_train = (df['price'] - df['price'].min()) / (df['price'].max() - df['price'].min())
X_train = df[['area', 'bedrooms', 'bathrooms']].values

class LinearRegression:
    def __init__(self, w=None, b=0):
        self.w = np.random.rand(X_train.shape[1]) if w is None else w
        self.b = b
    
    def cost(self, X, y):
        num = len(X)
        predictions = X.dot(self.w) + self.b
        return (1 / (2 * num)) * np.sum((predictions - y) ** 2)
    
    def gradient(self, X, y):
        num = len(X)
        predictions = X.dot(self.w) + self.b
        dw = (1 / num) * X.T.dot(predictions - y)
        db = (1 / num) * np.sum(predictions - y)
        return dw, db
    
    def train(self, X, y, learning_rate=0.01, iterations=10000):
        cost_history = []  # List to store the cost at each iteration
        for i in range(iterations):
            dw, db = self.gradient(X, y)
            self.w -= learning_rate * dw
            self.b -= learning_rate * db
            cost = self.cost(X, y)
            cost_history.append(cost)
            if i % 1000 == 0:
                print(f"Iteration {i}: cost = {cost}, w = {self.w}, b = {self.b}")
        self.plot_cost(cost_history)  # Plot cost after training
        return self.w, self.b
    
    def plot_cost(self, cost_history):
        plt.plot(range(len(cost_history)), cost_history, color='purple')
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Cost Function over Iterations")
        plt.show()

    def predict(self, X):
        return X.dot(self.w) + self.b
    
    def plot(self, X, y):
        plt.scatter(df['area'], y, color='red', label="Actual Data")
        # Use the average values for 'bedrooms' and 'bathrooms'
        area_range = np.linspace(df['area'].min(), df['area'].max(), 100).reshape(-1, 1)
        bedrooms_avg = np.full_like(area_range, df['bedrooms'].mean())
        bathrooms_avg = np.full_like(area_range, df['bathrooms'].mean())
        X_plot = np.hstack((area_range, bedrooms_avg, bathrooms_avg))
        y_pred = self.predict(X_plot)
        plt.plot(area_range, y_pred, color="blue", label="Regression Line")
        plt.xlabel("Normalized Area")
        plt.ylabel("Normalized Price")
        plt.legend()
        plt.show()

# Instantiate and train the model
model = LinearRegression()
model.train(X_train, y_train)
model.plot(X_train, y_train)
