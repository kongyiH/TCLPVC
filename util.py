import numpy as np
import random
import torch
from sklearn import manifold
from matplotlib import pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances


def set_seed(seed_num):
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)
    torch.backends.cudnn.deterministic = True


def getMvKNNGraph(X, k=5, mode='connectivity'):
    MvG = []
    for x in X:
        subG = kneighbors_graph(x, n_neighbors=k, n_jobs=-1, include_self=False, mode=mode)
        subG = subG.toarray()

        MvG.append(subG)
    return np.array(MvG)


def euclidean_dist(x, y, root=False):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """

    m, n = x.size(0), y.size(0)

    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy

    dist.addmm_(1, -2, x, y.t())
    if root:
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def calculate_cosine_similarity(x1, x2):
    from sklearn.preprocessing import normalize
    x1_ = normalize(x1, axis=1)
    x2_ = normalize(x2, axis=1)
    similarity = np.matmul(x1_, x2_.T)
    return similarity


def calculate_degree_matrix(similarity_matrix):
    degree_matrix = torch.diag(torch.sum(similarity_matrix, dim=1))
    return degree_matrix


def calculate_laplacian(similarity_matrix, k=10):
    similarity_matrix = (similarity_matrix + similarity_matrix.t()) * 0.5
    if k > 0:
        similarity_matrix = knn(similarity_matrix, k=k)
    similarity_matrix = (similarity_matrix + similarity_matrix.t()) * 0.5
    # 2. 计算度矩阵
    degree_matrix = calculate_degree_matrix(similarity_matrix)

    # 3.计算拉普拉斯矩阵
    laplacian_matrix = degree_matrix - similarity_matrix

    return laplacian_matrix


def normalize(x):
    """Normalize"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


def normalize_row(x):
    x = (x - np.tile(np.min(x, axis=0), (x.shape[0], 1))) / np.tile((np.max(x, axis=0) - np.min(x, axis=0)),
                                                                    (x.shape[0], 1))
    return x


def compute_knn_graphs(X, M, k):
    """
    计算每个视图上存在的部分实例的k近邻图。

    :param X: 包含v个视图的列表，每个视图是一个N*d的np.array。
    :param M: 掩码矩阵，N*v的矩阵，每个元素只有0，1两个值。
    :param k: 近邻数量。
    :return: 包含v个N*N矩阵的列表，矩阵为0表示不是k近邻，为1表示是k近邻。
    """
    v = len(X)  # 视图的数量
    N = X[0].shape[0]  # 样本数量
    knn_graphs = []  # 存储每个视图的k近邻图

    for i in range(v):
        # 计算距离矩阵
        dist_matrix = pairwise_distances(X[i], X[i])

        # 初始化k近邻图为全0矩阵
        knn_graph = np.zeros((N, N))

        for j in range(N):
            # 如果样本在当前视图中存在
            if M[j, i] == 1:
                # 获取距离并进行排序，返回索引
                dists = dist_matrix[j]
                neighbors_idx = np.argsort(dists)

                # 选取k个最近邻（除去自身，所以是从1开始）
                for neighbor in neighbors_idx[1:k + 1]:
                    # 在k近邻图中标记为1
                    knn_graph[j, neighbor] = 1

        # 添加当前视图的k近邻图到结果列表
        knn_graphs.append(knn_graph)

    return np.array(knn_graphs)


def plot_embedding_2d(data, labels):
    # 标签到颜色的映射
    color_map = {1: 'b', 2: 'c', 3: 'olive', 4: 'peru', 5: 'm',
                 6: 'coral', 7: 'darkgreen', 8: 'deeppink', 9: 'gold', 10: 'lime',
                 11: 'maroon', 12: 'sienna', 13: 'steelblue', 14: 'yellowgreen', 15: 'whitesmoke',
                 16: 'aliceblue', 17: 'aquamarine', 18: 'lavender', 19: 'cadetblue', 20: 'cornflowerblue'}

    # 绘制散点图
    for i in range(len(data)):
        x, y = data[i]
        label = labels[i]
        plt.scatter(x, y, c=color_map.get(label, 'black'))

    # 添加图例
    # plt.legend()
    plt.show()


def plot_tsne(x, labels, plot=True):
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_tsne = tsne.fit_transform(x)

    if plot:
        plot_embedding_2d(x_tsne, labels)

    return x_tsne


def construct_category_matrix(label):
    num_samples = label.shape[0]

    category_matrix = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            category_matrix[i][j] = (label[i] == label[j])

    return category_matrix


def knn(matrix, k=10, largest=True):
    # 取出每一行前k个最大值的索引
    _, indices = torch.topk(matrix, k=k, dim=1, largest=largest, sorted=True)

    # 将其他元素置零
    mask = torch.zeros_like(matrix)
    mask.scatter_(1, indices, 1)
    matrix *= mask
    matrix = matrix - torch.diag(torch.diag(matrix))
    # m1 = matrix.cpu().numpy()
    # 返回保留前k个元素后的矩阵
    return matrix


# for data set
def generate_subsets(vector):
    """生成一个向量的所有非空子集。"""
    subsets = []
    n = len(vector)
    for i in range(1, 2 ** n):
        subset = [vector[j] if (i >> j) & 1 else 0 for j in range(n)]
        subsets.append(subset)
    return subsets


def is_subset(vector_a, vector_b):
    """检查vector_a是否是vector_b的子集。"""
    return all(a <= b for a, b in zip(vector_a, vector_b))


def generate_subset_pairs(vector):
    """生成所有有效的子集对并去除重复项。"""
    subsets = generate_subsets(vector)
    subset_pairs_set = set()

    for subset_a in subsets:
        for subset_b in subsets:
            # 确保subset_a是subset_b的子集，且它们不相等，排除全0向量。
            if is_subset(subset_a, subset_b) and subset_a != subset_b and any(subset_a):
                pair = (tuple(subset_a), tuple(subset_b))  # 将列表转换为元组
                subset_pairs_set.add(pair)

    # 将每个子集对的元组转换回列表
    subset_pairs_list = [(list(pair[0]), list(pair[1])) for pair in subset_pairs_set]

    return subset_pairs_list


def average_similarity(g_f, mask):
    """
    直接在g_f求和得到N*N的矩阵g_sum，然后根据mask按行除以可见视图的总和。

    参数:
    - g_f: 一个列表，包含v个元素，每个元素是一个N*N的numpy数组，表示指定视图的相似性矩阵。
    - mask: 一个N*v的numpy数组，表示样本在指定视图上是否缺失，1表示存在，0表示缺失。

    返回:
    - g_sum_normalized: 归一化后的总相似性矩阵。
    """
    # 初始化总相似性矩阵
    N = mask.shape[0]
    g_sum = np.zeros((N, N))

    # 累加每个视图的相似性矩阵
    for g in g_f:
        g_sum += g

    # 计算每个样本的可见视图总和
    visible_counts = np.sum(mask, axis=1)
    visible_counts[visible_counts == 0] = 1  # 防止除以0

    # 将可见视图总和扩展为N*N的矩阵，用于按元素归一化g_sum
    visible_counts_matrix = np.outer(visible_counts, np.ones(N))

    # 按行归一化g_sum
    g_sum_normalized = g_sum / visible_counts_matrix

    return g_sum_normalized


def knn_numpy(g, k):
    """
    根据相似性矩阵g计算k近邻。
    g: N*N的相似性矩阵
    k: 每个节点的k近邻数量
    返回一个N*N的0和1矩阵，表示k近邻。
    """
    N = g.shape[0]
    # 将自身的相似性设置为最小，确保不会选为自己的近邻
    np.fill_diagonal(g, -np.inf)

    # 初始化0和1的近邻矩阵
    knn_mat = np.zeros((N, N), dtype=int)

    # 找到每个样本的k个最近邻居
    # 注意，这里我们使用np.argsort而不是np.argpartition，因为相似性越大越好
    nearest_indices = np.argsort(-g, axis=1)[:, :k]

    # 遍历每个样本，标记其k个最近邻居
    for i in range(N):
        knn_mat[i, nearest_indices[i]] = 1

    return knn_mat


def recover_missing_expressions(G, z_all, mask):
    """
    G: N*N的k近邻图，值为1表示是k近邻，为0表示不是。
    z_all: 含有v个视图特征信息的list，每个视图都是一个N*d的矩阵，表示N个样本的特征。
    mask: N*v的掩码矩阵，表示z_all在指定样本的视图上是否存在缺失，为0表示实例缺失，为1表示存在。
    """
    N, v = mask.shape
    recovered_z_all = [np.copy(z) for z in z_all]  # 复制一份z_all以避免修改原始数据

    for view_idx in range(v):  # 遍历每个视图
        z = z_all[view_idx]
        for i in range(N):  # 遍历每个样本
            if mask[i, view_idx] == 0:  # 如果样本i在视图view_idx上缺失
                neighbors = np.where(G[i] == 1)[0]  # 找到样本i的k近邻
                visible_neighbors = [n for n in neighbors if mask[n, view_idx] == 1]  # 过滤出可见的近邻

                if visible_neighbors:  # 如果有可见的近邻
                    # 计算可见近邻的平均表达并赋值给缺失实例
                    recovered_z_all[view_idx][i] = np.mean(z[visible_neighbors], axis=0)

    return recovered_z_all
