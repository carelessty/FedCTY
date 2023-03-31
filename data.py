#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from libs import *

"""
-------
CIFAR
Dirichilet distribution
----
"""


def partition_CIFAR_dataset(
    dataset,
    file_name: str,
    balanced: bool,
    matrix,
    n_clients: int,
    n_classes: int,
    train: bool,
):
    """Partition dataset into `n_clients`.
    Each client i has matrix[k, i] of data of class k"""

    list_clients_X = [[] for i in range(n_clients)]
    list_clients_y = [[] for i in range(n_clients)]

    if balanced and train:
        n_samples = [500] * n_clients
    elif balanced and not train:
        n_samples = [100] * n_clients
    elif not balanced and train:
        n_samples = (
            [100] * 2 + [250] * 10 + [500] * 10 + [750] * 5 + [1000] * 3 + [100] * 8 + [250] * 20 + [500] * 20 + [
                750] * 15 + [1000] * 7
            )
    elif not balanced and not train:
        n_samples = [20] * 2 + [50] * 10 + [100] * 10 + [150] * 5 + [200] * 3 + [20] * 8 + [50] * 20 + [100] * 20 + [
            150] * 15 + [200] * 7

    list_idx = []
    for k in range(n_classes):

        idx_k = np.where(np.array(dataset.targets) == k)[0]
        list_idx += [idx_k]             # 第一维：标签值 第二维：属于该标签的条目下标

    for idx_client, n_sample in enumerate(n_samples):
        # 客户下标    客户样本数

        clients_idx_i = []  # client_i 分到的数据条目下标
        client_samples = 0

        for k in range(n_classes):

            if k < n_classes - 1:
                samples_digit = int(matrix[idx_client, k] * n_sample)
            if k == n_classes - 1:
                samples_digit = n_sample - client_samples
            client_samples += samples_digit

            clients_idx_i = np.concatenate(
                # 将标签k的数据随机选取分给client_i
                (clients_idx_i, np.random.choice(list_idx[k], samples_digit))
            )

        # clients_idx_i 当前客户所持有数据 在数据集中的下标
        clients_idx_i = clients_idx_i.astype(int)

        for idx_sample in clients_idx_i:

            # 客户idx_client 数据样本
            list_clients_X[idx_client].append(dataset.data[idx_sample])
            # 客户idx_client 数据标签
            list_clients_y[idx_client].append(dataset.targets[idx_sample])

        list_clients_X[idx_client] = np.array(list_clients_X[idx_client])

    folder = "./data/"
    with open(folder + file_name, "wb") as output:
        pickle.dump((list_clients_X, list_clients_y), output)


def create_CIFAR10_dirichlet(
    dataset_name: str,
    balanced: bool,
    alpha: float,
    n_clients: int,
    n_classes: int,
):
    """ 根据Dir(alpha)创建CIFAR-10数据集"""

    from numpy.random import dirichlet

    # matrix = dirichlet([alpha] * n_classes, size=n_clients)
    matrix = np.concatenate((dirichlet([alpha * 100] * n_classes, size=20),
                             dirichlet([alpha * 10000] * n_classes, size=10),
                             dirichlet([alpha] * n_classes, size=70)))

    CIFAR10_train = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    CIFAR10_test = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    file_name_train = f"{dataset_name}_train_{n_clients}.pkl"
    partition_CIFAR_dataset(
        CIFAR10_train,
        file_name_train,
        balanced,
        matrix,
        n_clients,
        n_classes,
        True,
    )

    file_name_test = f"{dataset_name}_test_{n_clients}.pkl"
    partition_CIFAR_dataset(
        CIFAR10_test,
        file_name_test,
        balanced,
        matrix,
        n_clients,
        n_classes,
        False,
    )


class CIFARDataset(Dataset):
    """Convert the CIFAR pkl file into a Pytorch Dataset"""

    def __init__(self, file_path: str, k: int):

        dataset = pickle.load(open(file_path, "rb"))

        if k is None:
            self.X = dataset[0][1]
            yt = np.array(dataset[1][1])
            self.y = torch.from_numpy(yt).type(torch.long)
        else:
            self.X = dataset[0][k]
            yt = np.array(dataset[1][k])
            self.y = torch.from_numpy(yt).type(torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):

        # 3D input 32x32x3
        x = torch.Tensor(self.X[idx]).permute(2, 0, 1) / 255
        x = (x - 0.5) / 0.5
        y = self.y[idx]

        return x, y


def clients_set_CIFAR(
    file_name: str, n_clients: int, batch_size: int, shuffle=True
):
    """Download for all the clients their respective dataset"""
    print("数据集文件已找到：" + file_name)

    list_dl = list()

    for k in range(n_clients):
        dataset_object = CIFARDataset(file_name, k)

        dataset_dl = DataLoader(
            dataset_object, batch_size=batch_size, shuffle=shuffle
        )

        list_dl.append(dataset_dl)

    return list_dl


"""
---------
Upload any dataset
Puts all the function above together
---------
"""


def get_dataloaders(dataset, batch_size: int, shuffle=True):

    folder = "./data/"

    if dataset == "CIFAR10_iid":
        n_clients = 100
        samples_train, samples_test = 500, 100

        CIFAR10_train = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        CIFAR10_train_split = torch.utils.data.random_split(
            CIFAR10_train, [samples_train] * n_clients
        )
        list_dls_train = [
            torch.utils.data.DataLoader(
                ds, batch_size=batch_size, shuffle=True)
            for ds in CIFAR10_train_split
        ]

        CIFAR10_test = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        CIFAR10_test_split = torch.utils.data.random_split(
            CIFAR10_test, [samples_test] * n_clients
        )
        list_dls_test = [
            torch.utils.data.DataLoader(
                ds, batch_size=batch_size, shuffle=True)
            for ds in CIFAR10_test_split
        ]

    elif dataset[:5] == "CIFAR" or dataset[:5] == "cifar":

        n_classes = 10          # 类别
        n_clients = 100         # client个数
        #balanced = dataset[8:12] == "bbal"      # 数据分布是否均衡
        balanced = "bbal"
        alpha = 0.01
        #alpha = float(dataset[13:])             # 迪利克雷分布alpha参数，越小越不均匀

        file_name_train = f"{dataset}_train_{n_clients}.pkl"
        path_train = folder + file_name_train

        file_name_test = f"{dataset}_test_{n_clients}.pkl"
        path_test = folder + file_name_test

        if not os.path.isfile(path_train):
            print("无数据集文件！creating dataset alpha:", alpha)
            create_CIFAR10_dirichlet(
                dataset, balanced, alpha, n_clients, n_classes
            )

        list_dls_train = clients_set_CIFAR(
            path_train, n_clients, batch_size, True
        )

        list_dls_test = clients_set_CIFAR(
            path_test, n_clients, batch_size, True
        )
        #clients_shannon = pickle.load(open(path_train, "rb"))[2]
        print(f"CIFAR10_nonIID数据集加载成功！")

    else:
        print("Dataset para Error!")
        import sys
        sys.exit()

    return list_dls_train, list_dls_test



