from perceptron.training import Perceptron
from sklearn.model_selection import train_test_split
import csv

def main():
    print("Perceptron training.")

    # Load dataset
    dataset = []
    while not dataset:
        data_path = input("Enter the file path for the dataset: ")
        try:
            with open(data_path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    dataset.append([float(x) for x in row])
        except FileNotFoundError:
            print("Invalid file path.")

    # Set up train_test_split
    X = [row[:-1] for row in dataset]
    y = [row[-1] for row in dataset]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
    test_data = [X_test[i] + [y_test[i]] for i in range(len(X_test))]

    print(f"Dataset split: {len(train_data)} training, {len(test_data)} testing samples.")

    # Set up and run perceptron
    num_inputs = len(X[0]) + 1  # +1 for bias
    weights = [1.0] * num_inputs
    p = Perceptron(num_inputs, weights)
    p.training(train_data)

    # Evaluate on test set
    correct = 0
    for row in test_data:
        inputs = [1.0] + row[:-1]  # Add bias
        actual = row[-1]
        prediction = p.activation(p.weighted_sum(inputs))
        if prediction == actual:
            correct += 1

    accuracy = correct / len(test_data) * 100 if test_data else 0
    print("Training completed.")
    print("Final weights:", p.weights)
    print(f"Test accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
