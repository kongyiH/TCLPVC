import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy.linalg as lg
import scipy.linalg as slg
import numpy as np
from scipy.optimize import linear_sum_assignment

from util import euclidean_dist, calculate_cosine_similarity, calculate_laplacian


# distance base method
def row2one(P):
    P_sum = P.sum(dim=1, keepdim=True)
    one = torch.ones(1, P.shape[1]).to(P.device)
    return P - (P_sum - 1).mm(one) / P.shape[1]


def col2one(P):
    P_sum = P.sum(dim=0, keepdim=True)
    one = torch.ones(P.shape[0], 1).to(P.device)
    return P - (one).mm(P_sum - 1) / P.shape[0]


def P_init(D):
    P = torch.zeros_like(D)
    D_rowmin = D.clone()
    max_d = D.max()
    min_ind = torch.argmin(D_rowmin, dim=0)
    D_rowmin[:, :] = max_d
    D_rowmin = D_rowmin.scatter(0, min_ind.unsqueeze(0), D[min_ind, torch.arange(D.shape[1]).long()].unsqueeze(0))

    _, idx_max = torch.min(D_rowmin, dim=1)

    P[torch.arange(D.shape[0]).long(), idx_max.long()] = 1.0

    return P


def PVC(D, tau_1=30, tau_2=10, lr=0.1):
    P = P_init(D)

    d = [torch.zeros_like(D) for _ in range(3)]

    for i in range(tau_1):
        P = P - lr * D
        for j in range(tau_2):
            P_0 = P.clone()
            P = P + d[0]
            Y = row2one(P)
            d[0] = P - Y

            P = Y + d[1]
            Y = col2one(P)
            d[1] = P - Y

            P = Y + d[2]
            Y = F.relu(P)
            d[2] = P - Y

            P = Y
            if (P - P_0).norm().item() == 0:
                break

    max_val, max_idx = torch.max(P, dim=1, keepdim=True)
    P_pred = torch.zeros_like(P)
    P_pred.scatter_(1, max_idx, 1)

    return P, P_pred


# GOT
class GOT(nn.Module):
    def __init__(self, nodes, tau=2, it=30):
        super(GOT, self).__init__()
        self._nodes = nodes
        self._tau = tau
        self._it = it
        self.mean = nn.Parameter(torch.rand((self._nodes, self._nodes), dtype=torch.float32), requires_grad=True)
        self.std = nn.Parameter(10 * torch.ones((self._nodes, self._nodes), dtype=torch.float32), requires_grad=True)

    def init_param(self, similarity):
        self.mean.data = similarity

    def doubly_stochastic(self, P):
        """Uses logsumexp for numerical stability."""
        A = P / self._tau
        for i in range(self._it):
            A = A - A.logsumexp(dim=1, keepdim=True)
            A = A - A.logsumexp(dim=0, keepdim=True)
        return torch.exp(A)

    def forward(self, eps):
        P_noisy = self.mean + self.std * eps
        DS = self.doubly_stochastic(P_noisy)
        return DS

    def loss_got(self, g1, g2, DS, params):
        [C1_tilde, C2_tilde] = params
        loss_c = torch.trace(g1) + torch.trace(DS @ g2 @ torch.transpose(DS, 0, 1))
        # svd version
        u, sigma, v = torch.svd(C2_tilde @ torch.transpose(DS, 0, 1) @ C1_tilde)
        loss = loss_c - 2 * torch.sum(sigma)
        return loss


def wasserstein_initialisation(A, B, device='cpu'):
    # Wasserstein directly on covariance
    Root_1 = slg.sqrtm(A)
    Root_2 = slg.sqrtm(B)
    C1_tilde = torch.from_numpy(Root_1.astype(np.float32)).to(device)
    C2_tilde = torch.from_numpy(Root_2.astype(np.float32)).to(device)
    return [C1_tilde, C2_tilde]


def regularise_and_invert(x, y, alpha, ones):
    x_reg = regularise_invert_one(x, alpha, ones)
    y_reg = regularise_invert_one(y, alpha, ones)
    return [x_reg, y_reg]

def regularise_invert_one(x, alpha, ones):
    if ones:
        x_reg = lg.inv(x + alpha * np.eye(len(x)) + np.ones([len(x), len(x)]) / len(x))
        # x_reg = torch.inverse(x + alpha * torch.eye(len(x)) + torch.ones([len(x), len(x)]) / len(x))
    else:
        x_reg = lg.pinv(x) + alpha * np.eye(len(x))
    return x_reg


def get_got_input(fea1, fea2, alpha=0.1, k=100, graph=True, device='cpu'):
    # 1. g1 and g2
    g1 = torch.from_numpy(calculate_cosine_similarity(fea1, fea1)).to(device)
    g2 = torch.from_numpy(calculate_cosine_similarity(fea2, fea2)).to(device)

    # 2. L1 and L2
    L1 = calculate_laplacian(g1, k=k).cpu().numpy()
    L2 = calculate_laplacian(g2, k=k).cpu().numpy()
    if graph:
        [L1_reg, L2_reg] = regularise_and_invert(L1, L2, alpha, ones=True)
    else:
        L1_reg = L1
        L2_reg = L2
    return g1, g2, L1_reg, L2_reg


def train_got(fea1, fea2, device):
    num_sample = fea1.shape[0]

    # get model and init
    model = GOT(num_sample)
    model.to(device)
    similarity = calculate_cosine_similarity(fea1, fea2)
    model.init_param(torch.from_numpy(similarity).to(device))

    # get optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5)

    # get got input
    g1, g2, L1_reg, L2_reg = get_got_input(fea1, fea2, k=20)

    # Initialization
    L1_tensor = torch.from_numpy(L1_reg.astype(np.float32)).to(device)
    L2_tensor = torch.from_numpy(L2_reg.astype(np.float32)).to(device)
    params = wasserstein_initialisation(L1_reg, L2_reg, device)
    history = []
    for epoch in range(200):
        cost = 0
        for iter in range(5):
            eps = torch.randn((model._nodes, model._nodes)).to(device)
            DS = model(eps)
            loss = model.loss_got(L1_tensor, L2_tensor, DS, params)
            cost += loss
        cost = cost / 5
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        history.append(cost.item())
        if epoch % 50 == 0:
            print('[Epoch %4d/200] loss: %f - std: %f' % (epoch, cost.item(), model.std.detach().mean()))
    P = model.doubly_stochastic(model.mean)
    max_val, max_idx = torch.max(P, dim=1, keepdim=True)
    P_pred = torch.zeros_like(P)
    P_pred.scatter_(1, max_idx, 1)
    return P.detach().cpu().numpy(), P_pred.detach().cpu().numpy()


def gm_dsn(X):
    n = len(X)

    for _ in range(30):
        x = X.copy()
        k1 = np.sum(X, axis=1) / n
        k2 = np.sum(X, axis=0) / n

        X = X + 1/n + np.sum(k1) / n

        for i in range(n):
            X[i, :] = X[i, :] - k2
            X[:, i] = X[:, i] - k1

        X = (X + np.abs(X)) / 2

    return X


def DSPFP(A1, A2):
    # compare graph
    n1, n2 = A1.shape[0], A2.shape[0]
    X = np.ones((n1, n2)) / n1

    ep = 100
    index = 1
    maxN = max(n1, n2)
    Y = np.zeros((maxN, maxN))

    a = 0.5
    obj = np.zeros(50)
    while ep >= 1e-6 and index <= 50:
        x = X.copy()
        Y[:n1, :n2] = A1 @ X @ A2
        Y = gm_dsn(Y)
        X = (1 - a) * X + a * Y[:n1, :n2]
        X = X / np.max(X)
        ep = np.max(np.abs(x - X))
        index += 1

    P = X
    A = np.zeros_like(P)
    for i in range(P.shape[1]):
        j = np.argmax(P[:, i])
        A[j, i] = 1
    P = A

    # 如果需要 dis_greedy 的实现，可以替换上述块的最后一行
    # P = dis_greedy(X)

    return P, P


def hungarian(fea1, fea2, device='cpu'):
    g = euclidean_dist(torch.from_numpy(fea1), torch.from_numpy(fea2)).numpy()

    # 使用linear_sum_assignment求解匈牙利算法
    row_idx, col_idx = linear_sum_assignment(g)

    # 创建P_pred矩阵
    P_pred = np.zeros_like(g)
    P_pred[row_idx, col_idx] = 1

    return P_pred, P_pred


def my_alignment(fea1, fea2, pre_labels1, pre_labels2, qs1, qs2, zs1, zs2, device='cpu'):
    g1 = calculate_cosine_similarity(fea1, fea2)
    # g = euclidean_dist(torch.from_numpy(fea1), torch.from_numpy(fea2)).numpy()

    # # g2 = calculate_cosine_similarity(qs1, qs2)
    # g2 = euclidean_dist(torch.from_numpy(qs1), torch.from_numpy(qs2))
    # g2, _ = PVC(g2)
    # g2 = g2.numpy()

    g = g1  # * g2

    # 使用linear_sum_assignment求解匈牙利算法
    row_idx, col_idx = linear_sum_assignment(-g)

    # 创建一个全零矩阵
    P = np.zeros_like(g)

    # 将最大值放置在对应位置
    P[row_idx, col_idx] = g[row_idx, col_idx]

    # 创建P_pred矩阵
    P_pred = np.zeros_like(g)
    P_pred[row_idx, col_idx] = 1

    return P, P_pred


def get_P(fea1, fea2, method, device='cpu', pre_labels1=None, pre_labels2=None, qs1=None, qs2=None, zs1=None, zs2=None):
    if method == 'PVC':
        dis = euclidean_dist(torch.from_numpy(fea1), torch.from_numpy(fea2))
        P, P_pred = PVC(dis)
    elif method == 'GOT':
        P, P_pred = train_got(fea1, fea2, device)
    elif method == 'DSPFP':
        g1 = calculate_cosine_similarity(fea1, fea1)
        g2 = calculate_cosine_similarity(fea2, fea2)
        np.fill_diagonal(g1, 0)
        np.fill_diagonal(g2, 0)
        P, P_pred = DSPFP(g1, g2)
    elif method == 'hun':
        P, P_pred = hungarian(fea1, fea2, device)
    else:
        P, P_pred = my_alignment(fea1, fea2, pre_labels1, pre_labels2, qs1, qs2, zs1, zs2)

    return P, P_pred


def get_alignment(fea, hs, qs, zs, pre_labels, gt_labels, aligned_idx, device='cpu'):
    """
    To compute CAR and IAR, the alignment data is directly input,
    and the alignment relations are recalculated by isolating
    and excluding the misaligned components using the alignment indices.
    """
    method = 'my'

    # Check if all data is already aligned
    if np.all(aligned_idx == 1):
        print("All data is already aligned.")
        return fea, gt_labels, aligned_idx

    # Step 1: 提取未对齐部分的数据
    unaligned_idx = (aligned_idx == 0)

    # 获取未对齐的数据
    fea_unaligned = [f[unaligned_idx] for f in fea]
    hs_unaligned = [h[unaligned_idx] for h in hs]
    qs_unaligned = [q[unaligned_idx] for q in qs]
    zs_unaligned = [z[unaligned_idx] for z in zs]
    pre_labels_unaligned = [pl[unaligned_idx] for pl in pre_labels]

    # Step 2: 计算对齐矩阵 P 和 P_pred
    P, P_pred = get_P(hs_unaligned[0], hs_unaligned[1], method=method, device=device,
                      pre_labels1=pre_labels_unaligned[0], pre_labels2=pre_labels_unaligned[1],
                      qs1=qs_unaligned[0], qs2=qs_unaligned[1], zs1=zs_unaligned[0], zs2=zs_unaligned[1])

    # Step 3: 根据 P_pred 重新对齐未对齐的数据
    fea_aligned = []
    for i in range(len(fea)):
        fea_aligned.append(np.copy(fea[i]))  # 复制一份fea，保留对齐部分
    fea_aligned[1][unaligned_idx] = np.dot(P_pred, fea_unaligned[1])

    # Step 4: 计算 IAR 和 CAR
    # IAR: 重新对齐后的数据是否与aligned_idx=1的部分一致
    pred_align_idx = np.argmax(P_pred, axis=1)  # 预测的对齐索引
    # IAR = np.mean(aligned_idx[unaligned_idx] == 1)  # 检查未对齐部分是否成功对齐
    #
    # # CAR: 重新对齐后的数据的两个视图在真实标签上是否属于同一类别
    # gt_labels_pred_view1 = gt_labels[0][unaligned_idx]  # 视图1的真实标签
    # gt_labels_pred_view2 = gt_labels[1][unaligned_idx][pred_align_idx]  # 根据预测对齐矩阵重新排列视图2
    # CAR = np.mean(gt_labels_pred_view1 == gt_labels_pred_view2)
    # print("CAR=" + str(CAR) + ", miss=" + str(gt_labels_pred_view2.shape[0] * (1-CAR)))

    gt_realigned_labels = [np.array(x) for x in gt_labels]
    gt_realigned_labels[1][unaligned_idx] = gt_realigned_labels[1][unaligned_idx][pred_align_idx]
    # CAR = np.mean(gt_realigned_labels[0] == gt_realigned_labels[1])

    # # Step 5: 计算 realigned_idx
    # # 对比重新对齐的两个视图的 pre_labels，判断是否属于相同类别
    # realigned_idx = np.copy(aligned_idx)
    # pre_labels_pred_view1 = pre_labels_unaligned[0]  # 视图1的预测标签
    # pre_labels_pred_view2 = pre_labels_unaligned[1][pred_align_idx]  # 根据预测对齐矩阵重新排列视图2的预测标签
    #
    # # 重新对齐后的样本标签如果相同，则标记为 1，否则为 0
    # realigned_idx[unaligned_idx] = (pre_labels_pred_view1 == pre_labels_pred_view2).astype(int)

    # Step 5: Update realigned_idx using smallest distances from qs
    realigned_idx = np.copy(aligned_idx)
    unaligned_distances = np.linalg.norm(qs_unaligned[0] - qs_unaligned[1][pred_align_idx], axis=1)

    # If fewer than 1/10 remain, align all remaining unaligned samples
    num_to_align = max(len(aligned_idx) // 10, 1)  # Number of samples to align each call
    if np.sum(unaligned_idx) <= num_to_align:
        realigned_idx[unaligned_idx] = 1
    else:
        # Align the 1/10 of samples with smallest distances
        threshold = np.partition(unaligned_distances, num_to_align - 1)[num_to_align - 1]
        new_aligned = unaligned_distances <= threshold
        realigned_idx[unaligned_idx] = new_aligned.astype(int)

    # realigned_idx = np.ones_like(aligned_idx)

    # 返回重新对齐的fea
    # import scipy.io as sio
    # sio.savemat('BDGP_me.mat', {'fea': fea, 'fea_re': fea_aligned, 'hs': hs, 'qs': qs, 'zs': zs, 'labels': gt_labels,
    #                             'labels_re': gt_realigned_labels, 'P': P_pred, 'aligned_idx': aligned_idx})
    return fea_aligned, [labels.tolist() for labels in gt_realigned_labels], realigned_idx
