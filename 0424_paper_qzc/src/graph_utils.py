import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import shortest_path
import scipy.io # 如果数据是 .mat 格式

# 假设 X 是加载的特征矩阵 (N x D)
# 假设 K = 10

# 1. 计算 K 近邻图 (连接性)
knn_graph_connectivity = kneighbors_graph(X, n_neighbors=K, mode='connectivity', include_self=False, n_jobs=-1)

# 2. 计算距离平方，用于高斯核
# 注意：直接计算所有 N*N 对距离可能非常耗内存，最好结合稀疏性
# 一个高效的方法是先计算 KNN，然后只计算邻居间的权重

# 获取 KNN 的索引和距离 (sklearn 1.2+)
# 或者使用更底层的 Annoy, Faiss 等库获取精确或近似的 KNN
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=K+1, algorithm='auto').fit(X) # K+1 因为包含自身
distances, indices = nbrs.kneighbors(X)

# 提取第 K 个邻居的距离 (索引 K, 因为 distances[i, 0] 是自身距离 0)
# 如果用 kneighbors_graph 计算的，需要另行计算第 K 邻居距离
dK_distances = distances[:, K] # Shape (N,)

# 3. 构建稀疏权重矩阵 W
n_samples = X.shape[0]
W_rows = []
W_cols = []
W_data = []

# 遍历每个样本及其邻居来计算高斯核权重
for i in range(n_samples):
    if dK_distances[i] == 0: # 防止除以零
        sigma_i_sq = 1.0 # 或者一个小的 epsilon
    else:
        sigma_i_sq = dK_distances[i]**2
        if sigma_i_sq == 0: # 再次检查
            sigma_i_sq = 1e-6

    for j_idx in range(1, K + 1): # 从 1 开始跳过自身
        j = indices[i, j_idx]
        dist_sq = np.sum((X[i] - X[j])**2)

        # 根据公式 (1) 计算权重
        weight = np.exp(-4 * dist_sq / sigma_i_sq)

        W_rows.append(i)
        W_cols.append(j)
        W_data.append(weight)

W = csr_matrix((W_data, (W_rows, W_cols)), shape=(n_samples, n_samples))

# 4. 对称化
W = (W + W.T) / 2

# 5. 对角线置零 (通常 W 已经是 0 了，但以防万一)
W.setdiag(0)
W.eliminate_zeros() # 清理小的零值

print(f"Graph W created: shape {W.shape}, non-zero elements {W.nnz}")

# 可选：计算测地线距离 (如果需要精确复现 NN)
# dist_matrix_geodesic = shortest_path(csgraph=W, directed=False, method='auto', unweighted=False)
# 注意：这可能非常慢且耗内存！仅在小图上可行或使用近似算法。