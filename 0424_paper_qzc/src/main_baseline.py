# src/main_baseline.py
import numpy as np
import graphlearning as gl # 导入 graphlearning
import time
from sklearn.semi_supervised import LabelPropagation # 或其他基线模型
# KNN (Feature-based) is removed as we don't load features X
from sklearn.metrics import accuracy_score

# --- Parameters ---
dataset = 'mnist'         # Options: 'mnist', 'fashion_mnist', 'cifar10'
labels_per_class = 1    # Number of labeled samples per class
# n_trials = 100            # Number of random trials
n_trials = 5 # 改为 5 进行测试
K = 10                    # Number of neighbors for the graph
n_classes = 10            # Number of classes in these datasets

# --- Determine the correct metric for graphlearning ---
if dataset == 'mnist':
    metric = 'vae_old' # Use the old VAE embedding as per Poisson paper README
elif dataset == 'fashion_mnist':
    metric = 'vae'     # Use the standard VAE embedding
elif dataset == 'cifar10':
    metric = 'aet'     # Use the AutoEncoding Transformations embedding
else:
    raise ValueError("Unknown dataset: %s" % dataset)

# --- Load Labels using graphlearning ---
print(f"Loading labels for {dataset}...")
try:
    Y = gl.datasets.load(dataset, labels_only=True)
    print(f"Labels Y loaded: shape {Y.shape}, type {type(Y)}")
except Exception as e:
    print(f"Error loading labels using graphlearning: {e}")
    exit()

# --- Build/Load Graph W using graphlearning ---
print(f"Building/Loading graph W for {dataset} (metric: {metric}, K={K})...")
start_graph_time = time.time()
try:
    # This function likely computes KNN on the specified embedding and applies Gaussian kernel
    W = gl.weightmatrix.knn(dataset, K, metric=metric, kernel='gaussian')
    print(f"Graph W loaded/built: shape {W.shape}, type {type(W)}, nnz={W.nnz}")
    print(f"Graph construction time: {time.time() - start_graph_time:.2f} seconds.")
except Exception as e:
    print(f"Error building/loading graph using graphlearning: {e}")
    exit()

# --- Prepare results storage ---
all_indices = np.arange(Y.shape[0])
results_lp = []
# results_knn_feat = [] # Removed, no features X

# --- Start Experiment Trials ---
print(f"\nStarting {n_trials} trials with {labels_per_class} labels per class...")
total_trials_time_start = time.time()

for trial in range(n_trials):
    trial_time_start = time.time()
    print(f"\nRunning Trial {trial + 1}/{n_trials}")

    # 1. Generate Training Set Indices using graphlearning
    #    'rate' parameter in generate corresponds to labels_per_class
    labeled_indices = gl.trainsets.generate(Y, rate=labels_per_class)

    # Ensure labeled_indices is a numpy array for boolean indexing later
    labeled_indices = np.array(labeled_indices)

    if len(labeled_indices) == 0:
        print("Warning: No labeled samples generated, skipping trial.")
        continue

    # Determine unlabeled indices
    is_labeled = np.zeros(Y.shape[0], dtype=bool)
    is_labeled[labeled_indices] = True
    unlabeled_indices = all_indices[~is_labeled]

    print(f"  Sampling done: {len(labeled_indices)} labeled, {len(unlabeled_indices)} unlabeled.")

    # Prepare labels for scikit-learn (-1 for unlabeled)
    y_train = np.full(Y.shape[0], -1, dtype=int)
    y_train[labeled_indices] = Y[labeled_indices]

    # Get true labels for the unlabeled set for evaluation
    y_true_unlabeled = Y[unlabeled_indices]

    # 2. Run Baseline: Laplace (Label Propagation)
    try:
        lp_model = LabelPropagation(n_jobs=-1) # 可以省略 kernel，或设为 'knn'
        lp_model.fit(W, y_train)
        # Predict labels for all nodes, then select unlabeled ones
        y_pred_all = lp_model.predict(W)
        y_pred_lp_unlabeled = y_pred_all[unlabeled_indices]

        acc_lp = accuracy_score(y_true_unlabeled, y_pred_lp_unlabeled)
        results_lp.append(acc_lp)
        print(f"  Label Propagation Accuracy: {acc_lp:.4f}")
    except Exception as e:
        print(f"  Error during Label Propagation: {e}")
        results_lp.append(np.nan) # Append NaN if error occurs


    # 3. Baseline: KNN (Feature-based) - Cannot be run without features X
    #    We will skip this baseline.

    print(f"  Trial {trial + 1} finished in {time.time() - trial_time_start:.2f} seconds.")


# --- Aggregate and Print Final Results ---
total_trials_time_end = time.time()
print("\n--- Final Results ---")
print(f"Dataset: {dataset}, Metric: {metric}, Labels per class: {labels_per_class}, K={K}")
print(f"Total time for {n_trials} trials: {total_trials_time_end - total_trials_time_start:.2f} seconds.")

# Filter out potential NaN values from results if any errors occurred
valid_results_lp = [r for r in results_lp if not np.isnan(r)]
if valid_results_lp:
    mean_lp = np.mean(valid_results_lp)
    std_lp = np.std(valid_results_lp)
    print(f"Laplace (Label Propagation): Mean Acc = {mean_lp:.4f} ({mean_lp*100:.2f}%)")
    print(f"Laplace (Label Propagation): Std Dev  = {std_lp:.4f}")
    print(f"  (Based on {len(valid_results_lp)} successful trials)")
else:
    print("Laplace (Label Propagation): No successful trials to report.")

print("\nKNN (Feature-based): Not computed (requires original features X).")
print("\nCompare these results with Table 1 in paper 2408.05419v1.pdf.")