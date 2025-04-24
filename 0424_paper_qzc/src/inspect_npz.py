import numpy as np
import os

# --- ！！请确保这里的路径相对于你运行脚本的位置是正确的！！ ---
# 通常，如果你在 src 目录下运行，路径应该是 '../data/features/...'
# 如果你在项目根目录 (0424_paper_qzc) 下运行，路径是 'data/features/...'
# 请根据你的实际情况调整 base_path
base_path = '../data/features' # 或者 'data/features'

try:
    # 检查特征文件
    feature_path = os.path.join(base_path, 'mnist', 'MNIST_vae_old.npz')
    print(f"Inspecting features file: {feature_path}")
    with np.load(feature_path) as data:
        print("Keys in feature file:", data.files) # 打印所有键

    print("-" * 20) # 分隔符

    # 检查标签文件
    label_path = os.path.join(base_path, 'mnist', 'MNIST_labels.npz')
    print(f"Inspecting labels file: {label_path}")
    with np.load(label_path) as data:
        print("Keys in label file:", data.files) # 打印所有键

except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
    print("Please double-check the base_path and ensure the files are downloaded.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")