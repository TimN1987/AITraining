from typing import List

class Perceptron:
    def __init__(self, num_inputs: int = 3, weights: List[float] = [1, 1, 1]):
        self.num_inputs = num_inputs
        self.weights = weights

    def weighted_sum(self, inputs: List[float]) -> float:
        weighted_sum = 0
        for i in range(self.num_inputs):
            weighted_sum += self.weights[i] * inputs[i]
        return weighted_sum
    
    def activation(self, weighted_sum: float) -> int:
        return 1 if weighted_sum >= 0 else -1
    
    def training(self, training_set: List[float]):
        found_line = False
        while not found_line:
            total_error = 0
            for row in training_set:
                inputs = [1.0] + row[:-1]
                prediction = self.activation(self.weighted_sum(inputs))
                actual = row[-1]
                error = actual - prediction
                total_error += abs(error)
                for i in range(self.num_inputs):
                    self.weights[i] += error * inputs[i]
            if total_error == 0:
                found_line = True