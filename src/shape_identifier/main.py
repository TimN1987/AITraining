import os
import numpy as np
from PIL import Image
from training.network import ShapeIdentifier


def load_and_preprocess_image(image_path, image_size=28):
    """Loads an image, converts to grayscale, resizes, and normalizes."""
    try:
        img = Image.open(image_path).convert("L")
        img = img.resize((image_size, image_size))
        img_array = np.array(img) / 255.0
        return img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def ask_shape_types():
    """Prompt user for shape types."""
    default_shapes = ["circle", "square", "triangle"]
    print("Available shapes: circle, triangle, square, rectangle, pentagon, hexagon, diamond, star.")
    user_input = input(f"Enter shape types separated by commas (default {default_shapes}): ").strip()
    if not user_input:
        return default_shapes
    return [s.strip().lower() for s in user_input.split(",") if s.strip()]


def main():
    print("==============================================")
    print("         POLYGON SHAPE IDENTIFIER")
    print("==============================================\n")

    shapes = ask_shape_types()
    print(f"Using shapes: {shapes}\n")

    identifier = ShapeIdentifier(shapes, device="cpu")

    if identifier.load():
        print("Model loaded successfully.\n")
    else:
        print("No saved model found. You may need to train one.\n")

    while True:
        print("Options:")
        print("1. Train / Retrain the model")
        print("2. Identify a shape from an image")
        print("3. Quit")
        choice = input("Select an option (1-3): ").strip()

        if choice == "1":
            try:
                epochs = int(input("Enter number of training epochs (default 5): ") or 5)
                batch_size = int(input("Enter batch size (default 32): ") or 32)
            except ValueError:
                epochs, batch_size = 5, 32

            print("\nTraining started...")
            identifier.train(epochs=epochs, batch_size=batch_size)
            print("\nTraining complete!")
            identifier.save()
            print("Model saved.\n")

        elif choice == "2":
            image_path = input("Enter the path to the image file: ").strip()
            if not os.path.exists(image_path):
                print("File not found. Try again.\n")
                continue

            img = load_and_preprocess_image(image_path)
            if img is None:
                continue

            prediction = identifier.predict(img)
            print(f"\nPredicted shape: {prediction}\n")

        elif choice == "3":
            print("Exiting...")
            break

        else:
            print("Invalid option. Please select 1, 2, or 3.\n")


if __name__ == "__main__":
    main()
