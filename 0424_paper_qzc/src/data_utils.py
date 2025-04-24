# src/data_utils.py
import numpy as np
import os

def load_data(dataset_name, data_dir='data/features'):
    """
    Loads features and labels for a specified dataset.

    Args:
        dataset_name (str): The name of the dataset ('mnist', 'fashion_mnist', 'cifar10').
        data_dir (str): The base directory where dataset subfolders are located.

    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): Feature matrix (N x D).
            - Y (np.ndarray): Label vector (N,).

    Raises:
        ValueError: If the dataset_name is unknown.
        FileNotFoundError: If required .npz files are not found.
        KeyError: If expected keys ('data', 'labels') are not found in the .npz files.
                  You might need to adjust the keys based on actual file contents.
    """
    # --- Define file names based on our findings ---
    feature_files = {
        'mnist': 'MNIST_vae_old.npz',
        'fashion_mnist': 'FashionMNIST_vae.npz',
        'cifar10': 'cifar_aet.npz'  # As found in KNNData
    }
    label_files = {
        'mnist': 'MNIST_labels.npz',
        'fashion_mnist': 'FashionMNIST_labels.npz',
        'cifar10': 'cifar10_labels.npz' # From Data folder
    }

    if dataset_name not in feature_files or dataset_name not in label_files:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported datasets: 'mnist', 'fashion_mnist', 'cifar10'")

    # --- Construct full paths ---
    feature_path = os.path.join(data_dir, dataset_name, feature_files[dataset_name])
    label_path = os.path.join(data_dir, dataset_name, label_files[dataset_name])

    print(f"Attempting to load features from: {feature_path}")
    print(f"Attempting to load labels from: {label_path}")

    # --- Load data with error handling ---
    try:
        # Load features
        with np.load(feature_path) as feature_data:
            # !!! Critical Assumption: Adjust 'data' key if necessary !!!
            if 'data' in feature_data:
                X = feature_data['data']
            elif 'features' in feature_data: # Common alternative
                 X = feature_data['features']
            elif 'X' in feature_data: # Another common alternative
                 X = feature_data['X']
            else:
                raise KeyError("Could not find feature data key ('data', 'features', 'X') in feature file.")

        # Load labels
        with np.load(label_path) as label_data:
            # !!! Critical Assumption: Adjust 'labels' key if necessary !!!
            if 'labels' in label_data:
                Y = label_data['labels']
            elif 'L' in label_data: # Common alternative from GraphLearning repo
                Y = label_data['L']
            elif 'y' in label_data: # Another common alternative
                 Y = label_data['y']
            else:
                raise KeyError("Could not find label data key ('labels', 'L', 'y') in label file.")

        # Ensure Y is a 1D array
        if Y.ndim > 1:
             Y = Y.squeeze() # Remove singleton dimensions if necessary
             if Y.ndim > 1:
                  print(f"Warning: Labels array Y has shape {Y.shape} after squeeze, expected 1D.")


        print(f"Successfully loaded {dataset_name}: X shape {X.shape}, Y shape {Y.shape}")
        if X.shape[0] != Y.shape[0]:
             print(f"Warning: Number of samples in X ({X.shape[0]}) does not match Y ({Y.shape[0]})!")

        return X, Y

    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure the following files exist:")
        print(f"  - {feature_path}")
        print(f"  - {label_path}")
        print("Please download them from the jwcalder/GraphLearning repository's KNNData and Data folders.")
        raise
    except KeyError as e:
        print(f"Error: Key {e} not found in the corresponding .npz file.")
        print("Please inspect the .npz files (e.g., using `np.load(filepath).files`) to find the correct keys for features and labels.")
        raise


# --- Example Usage (Commented Out) ---
if __name__ == '__main__':
    try:
        print("Testing MNIST load...")
        X_mnist, Y_mnist = load_data('mnist', data_dir='../data/features') # Adjust path if needed

        print("\nTesting FashionMNIST load...")
        X_fashion, Y_fashion = load_data('fashion_mnist', data_dir='../data/features') # Adjust path if needed

        print("\nTesting CIFAR-10 load...")
        X_cifar, Y_cifar = load_data('cifar10', data_dir='../data/features') # Adjust path if needed

    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")