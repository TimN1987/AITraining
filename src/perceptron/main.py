from perceptron.training import Perceptron
import csv

def main():
    print("Perceptron training.")

    # Load training set
    data_loaded = False
    training_set = []
    while not data_loaded:
        data_path = input("Enter the file path for the training set: ")
        try:
            with open(data_path, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in reader:
                    training_set.append(row)
            data_loaded = True
        except:
            print('Invalid file path.')

    # Set up perceptron
    num_inputs = input("Enter number of inputs in dataset: ")
    num_inputs = int(num_inputs)
    weights = []
    for _ in range(num_inputs):
        weights.append(1)
    p = Perceptron(num_inputs, weights)
    p.training(training_set)
    print("Training completed.")
    print("Printing weights...")
    print(p.weights)

if __name__ == "__main__":
    main()