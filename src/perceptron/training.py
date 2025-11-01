from typing import List

class Perceptron:
    def __init__(self, num_inputs: int = 3, weights: List[int] = [1, 1, 1]):
        self.num_inputs = num_inputs
        self.weights = weights

    def weighted_sum(self, inputs):
        weighted_sum = 0
        for i in range(self.num_inputs):
            weighted_sum += self.weights[i] * inputs[i]
        return weighted_sum
    
    def activation(self, weighted_sum):
        return 1 if weighted_sum >= 0 else -1
    
    def training(self, training_set):
        found_line = False
        while not found_line:
            total_error = 0
            for inputs in training_set:
                prediction = self.activation(self.weighted_sum(inputs))
                actual = training_set[inputs]
                error = actual - prediction
                total_error += abs(error)
                for i in range(self.num_inputs):
                    self.weights[i] += error * inputs[i]
            if total_error == 0:
                found_line = True