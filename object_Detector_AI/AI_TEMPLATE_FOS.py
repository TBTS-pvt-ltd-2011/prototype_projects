
from ast import In
from itertools import product
from turtle import dot


In [1]: input_vector = [1.72, 1.23]
In [2]: weights_1 = [1.26, 0]
In [3]: weights_2 = [2.17, 0.32]

In [4]: # Computing the dot product of input_vector and weights_1
In [5]: first_indexes_mult = input_vector[0] * weights_1[0]
In [6]: second_indexes_mult = input_vector[1] * weights_1[1]
In [7]: dot_product_1 = first_indexes_mult + second_indexes_mult

In [8]: print(f"The dot product is: {dot_product_1}")
Out[8]: The dot product is: 2.1672

In [9]: import numpy as np

In [10]: dot_product_1 = np.dot(input_vector, weights_1)

In [11]: print(f"The dot product is: {dot_product_1}")
Out[11]: The dot product is: 2.1672

In [10]: dot_product_2 = np.dot(input_vector, weights_2)

In [11]: print(f"The dot product is: {dot_product_2}")
Out[11]: The dot product is: 4.1259

In [12]: # Wrapping the vectors in NumPy arrays
In [13]: input_vector = np.array([1.66, 1.56])
In [14]: weights_1 = np.array([1.45, -0.66])
In [15]: bias = np.array([0.0])

In [16]: def sigmoid(x):
   ...:     return 1 / (1 + np.exp(-x))

In [17]: def make_prediction(input_vector, weights, bias):
   ...:      layer_1 = np.dot(input_vector, weights) + bias
   ...:      layer_2 = sigmoid(layer_1)
   ...:      return layer_2

In [18]: prediction = make_prediction(input_vector, weights_1, bias)

In [19]: print(f"The prediction result is: {prediction}")
Out[19]: The prediction result is: [0.7985731]

In [20]: # Changing the value of input_vector
In [21]: input_vector = np.array([2, 1.5])

In [22]: prediction = make_prediction(input_vector, weights_1, bias)

In [23]: print(f"The prediction result is: {prediction}")
Out[23]: The prediction result is: [0.87101915]

In [24]: target = 0

In [25]: mse = np.square(prediction - target)

In [26]: print(f"Prediction: {prediction}; Error: {mse}")
Out[26]: Prediction: [0.87101915]; Error: [0.7586743596667225]

In [27]: derivative = 2 * (prediction - target)

In [28]: print(f"The derivative is {derivative}")
Out[28]: The derivative is: [1.7420383]

In [29]: # Updating the weights
In [30]: weights_1 = weights_1 - derivative

In [31]: prediction = make_prediction(input_vector, weights_1, bias)

In [32]: error = (prediction - target) ** 2

In [33]: print(f"Prediction: {prediction}; Error: {error}")
Out[33]: Prediction: [0.01496248]; Error: [0.00022388]

#the following code will shift to the chain rule()

derror_dweights = (
    derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
)

#the following code will terminate the chain rule() and end it

In [36]: def sigmoid_deriv(x):
   ...:     return sigmoid(x) * (1-sigmoid(x))

In [37]: derror_dprediction = 2 * (prediction - target)
In [38]: layer_1 = np.dot(input_vector, weights_1) + bias
In [39]: dprediction_dlayer1 = sigmoid_deriv(layer_1)
In [40]: dlayer1_dbias = 1

In [41]: derror_dbias = (
   ...:     derror_dprediction * dprediction_dlayer1 * dlayer1_dbias

class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )
        derror_dweights = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
            derror_dweights * self.learning_rate
        ) 

        In [42]: learning_rate = 0.1

In [43]: neural_network = NeuralNetwork(learning_rate)

In [44]: neural_network.predict(input_vector)
Out[44]: array([0.79412963])

class NeuralNetwork:
    # ...

    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors

        In [45]: # Paste the NeuralNetwork class code here
   ...: # (and don't forget to add the train method to the class)

In [46]: import matplotlib.pyplot as plt

In [47]: input_vectors = np.array(
   ...:     [
   ...:         [3, 1.5],
   ...:         [2, 1],
   ...:         [4, 1.5],
   ...:         [3, 4],
   ...:         [3.5, 0.5],
   ...:         [2, 0.5],
   ...:         [5.5, 1],
   ...:         [1, 1],
   ...:     ]
   ...: )

In [48]: targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])

In [49]: learning_rate = 0.1

In [50]: neural_network = NeuralNetwork(learning_rate)

In [51]: training_error = neural_network.train(input_vectors, targets, 10000)

In [52]: plt.plot(training_error)
In [53]: plt.xlabel("Iterations")
In [54]: plt.ylabel("Error for all training instances")
In [54]: plt.savefig("cumulative_error.png")