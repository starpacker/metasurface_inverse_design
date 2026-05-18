# data_collection
import numpy as np
import argparse
import torch
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime
from MetasurfaceDataLoad import SpecDataset


def get_loader(args):
    x_train, x_val,y_train, y_val = load_data()

    train_set = SpecDataset(
        struct=x_train,
        spectrum=y_train)

    test_set = SpecDataset(
        struct=x_val,
        spectrum=y_val)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=True
    )
    return train_data_loader, test_data_loader

def load_data():
    Stru_m1_du_c1 = np.load("D:/subject/physics/AI4S/project1/FDTDModel300/metasurface/collect_stru_1.npy")
    Stru_m1_du_c2 = np.load("D:/subject/physics/AI4S/project1/FDTDModel300/metasurface/collect_stru_2.npy")
    Stru_m1_du_c3 = np.load("D:/subject/physics/AI4S/project1/FDTDModel300/metasurface/collect_stru_3.npy")
    Stru_m1_du_c4 = np.load("D:/subject/physics/AI4S/project1/FDTDModel300/metasurface/collect_stru_4.npy")

    spec_du_c1 = np.load("D:/subject/physics/AI4S/project1/FDTDModel300/metasurface/collect_spec_1.npy")
    spec_du_c2 = np.load("D:/subject/physics/AI4S/project1/FDTDModel300/metasurface/collect_spec_2.npy")
    spec_du_c3 = np.load("D:/subject/physics/AI4S/project1/FDTDModel300/metasurface/collect_spec_3.npy")
    spec_du_c4 = np.load("D:/subject/physics/AI4S/project1/FDTDModel300/metasurface/collect_spec_4.npy")

    Bars3_t_cd = np.concatenate((spec_du_c1, spec_du_c2, spec_du_c3, spec_du_c4), axis=0)
    Bars3_pattern = np.concatenate((Stru_m1_du_c1, Stru_m1_du_c2, Stru_m1_du_c3, Stru_m1_du_c4), axis=0)

    del Stru_m1_du_c1, Stru_m1_du_c2, Stru_m1_du_c3, Stru_m1_du_c4
    del spec_du_c1, spec_du_c2, spec_du_c3, spec_du_c4

    bo_sum_cd = - np.abs(np.sum(Bars3_t_cd, axis=1))
    index_cd = np.argsort(bo_sum_cd)
    Num_of_chosen = 1000
    print(index_cd.shape[0])
    #Num_of_chosen = index_cd.shape[0]
    index_cd = index_cd[:Num_of_chosen].reshape(Num_of_chosen, 1)
    Bars3_t_cd = Bars3_t_cd[index_cd].reshape(-1, 60)

    Bars3_pattern = Bars3_pattern[index_cd, :, :].reshape(-1, 40, 40)


    # 对结构pattern输入进行数据扩充
    duplic1 = np.rot90(Bars3_pattern, k=-1, axes=(1, 2))
    duplic2 = np.rot90(Bars3_pattern, k=-2, axes=(1, 2))
    duplic3 = np.rot90(Bars3_pattern, k=-3, axes=(1, 2))
    duplic4 = np.flip(Bars3_pattern, 1)
    duplic5 = np.rot90(duplic4, k=-1, axes=(1, 2))
    duplic6 = np.rot90(duplic4, k=-2, axes=(1, 2))
    duplic7 = np.rot90(duplic4, k=-3, axes=(1, 2))
    pattern = np.vstack((Bars3_pattern, duplic1, duplic2, duplic3, duplic4, duplic5, duplic6, duplic7))
    pattern = pattern.reshape(-1, 1, 40, 40)

    del Bars3_pattern, duplic1, duplic2, duplic3, duplic4, duplic5, duplic6, duplic7

    # 对 t_lcp / t_rcp / cd 进行数据扩充
    Bars3_t_cd2 = (-1) * Bars3_t_cd
    cd = np.vstack((Bars3_t_cd, Bars3_t_cd, Bars3_t_cd, Bars3_t_cd, Bars3_t_cd2, Bars3_t_cd2, Bars3_t_cd2, Bars3_t_cd2))


    x_train, x_val, y_train, y_val = train_test_split(pattern, cd, test_size=0.1)
    return x_train, x_val, y_train, y_val

def parser_define():
    parser = argparse.ArgumentParser(description='CSNN Metasurface')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=512, type=int, help='batch size')
    parser.add_argument('-epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, default='./Fashion MNIST', help='root dir of Fashion-MNIST dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-opt', default='adam', type=str, help='use which optimizer. SDG or Adam')
    parser.add_argument('-momentum', default=0.999, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')
    parser.add_argument('-save-es', default=None,
                        help='dir for saving a batch spikes encoded by the first {Conv2d-BatchNorm2d-IFNode}')

    args = parser.parse_args()
    return args

def seed_define(seed = 3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
def plot_CD(net, train_set, args):
    img_inx = np.random.randint(1, 8000)
    img = train_set[img_inx][0]
    img = img.to(args.device)
    img = img.unsqueeze(0)
    y_pred = net(img)
    plt.figure(figsize = (12, 8))
    plt.subplot(1, 3, 1)
    plt.plot(train_set[img_inx][1])
    plt.title('label')
    plt.subplot(1, 3, 2)
    plt.plot(y_pred[0].cpu().detach().numpy())
    plt.title('predict')
    plt.subplot(1, 3, 3)
    plt.imshow(img.cpu().reshape(40,40), cmap=cm.Blues, alpha = 1)
    plt.title('structure')

def plot_weight_change(weight_changes):
    change_rates = []
    for i in range(1, len(weight_changes)):
        change_rate = {}
        for name, change in weight_changes[i].items():
            prev_change = weight_changes[i - 1][name]
            rate = (change - prev_change) / prev_change if prev_change != 0 else 0
            change_rate[name] = rate
        change_rates.append(change_rate)
    # 绘制权重变化率的折线图
    names = list(weight_changes[0].keys())
    epochs = list(range(1, len(change_rates) + 1))

    plt.figure(figsize=(12, 8))
    for i, name in enumerate(names):
        rates = [change_rate[name] for change_rate in change_rates]
        plt.plot(epochs, rates, label=name)

    plt.xlabel('Epochs')
    plt.ylabel('Weight Change Rate')
    plt.legend(loc='upper right', fontsize=8, ncol=4, frameon=False, bbox_to_anchor=(1.2, 1.5))
    plt.title('Weight Change Rate Across Epochs')
    plt.grid(True)
    plt.show()
    plt.savefig('./logs/Weight Change Rate Across Epochs' + datetime.datetime.now() + '.png')

def plot_lr(args, learning_rates):
    epochs = list(range(1, args.epochs + 1))
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, learning_rates, marker='.', linestyle='-', alpha=0.8)
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Change Across Epochs')
    plt.grid(True)
    plt.show()


