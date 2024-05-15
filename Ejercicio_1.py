import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, max_epochs=1000):
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]  # dot product + bias
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels):
        for _ in range(self.max_epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

            if all(self.predict(inputs) == label for inputs, label in zip(training_inputs, labels)):
                break

def plot_data_and_line(X, y, weights):
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.xlabel('X1')
    plt.ylabel('X2')

    if weights[1] != 0:
        slope = -weights[1] / weights[2]
        intercept = -weights[0] / weights[2]
        x_vals = np.array(plt.gca().get_xlim())
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, '--r', label='Separating line')

    plt.legend()
    plt.show()

def main():
    # Lectura de datos de entrenamiento y prueba
    train_data = pd.read_csv('OR_trn.csv', header=None)
    test_data = pd.read_csv('OR_tst.csv', header=None)

    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    # Entrenamiento del perceptrón
    perceptron = Perceptron(input_size=X_train.shape[1])
    perceptron.train(X_train, y_train)

    # Prueba del perceptrón entrenado
    correct_predictions = 0
    for inputs, label in zip(X_test, y_test):
        prediction = perceptron.predict(inputs)
        correct_predictions += 1 if prediction == label else 0
    accuracy = correct_predictions / len(y_test)
    print("Accuracy:", accuracy)

    # Visualización gráfica
    plot_data_and_line(X_train, y_train, perceptron.weights)

if __name__ == "__main__":
    main()
