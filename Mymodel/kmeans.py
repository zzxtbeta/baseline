import torch

def kmeans(x, attacker, ncluster, k, niter=20):
    '''
    x : torch.tensor(data_num,data_dim)
    attacker_center : torch.tensor(num_attackers, data_dim)
    ncluster : The number of clustering for data_num
    k : find the Top k clusters which are farthest away from attacker cluster
    niter : Number of iterations for kmeans
    '''

    N, D = x.size()

    c = x[torch.randperm(N)[:ncluster]] # init clusters at random
    for i in range(niter):
        a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)
        c = torch.stack([x[a==k].mean(0) for k in range(ncluster)])
        nanix = torch.any(torch.isnan(c), dim=1)
        ndead = nanix.sum().item()
        c[nanix] = x[torch.randperm(N)[:ndead]]

    cluster_indices = [torch.nonzero(a == k, as_tuple=True)[0] for k in range(ncluster)]
    attacker_center = attacker.mean(dim=0)

    # 根据与攻击者中心的距离从远到近排序
    distances = ((c - attacker_center[None, :])**2).sum(-1)
    sorted_indices = torch.argsort(distances, descending=True)

    c = c[sorted_indices]  # [32,128]
    cluster_indices = [cluster_indices[i] for i in sorted_indices]

    # 选择前k个最远的聚类中心
    normal_u_cluster = cluster_indices[:k]

    return normal_u_cluster




