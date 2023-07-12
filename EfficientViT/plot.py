import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.collections import PolyCollection
import torch

def sub_plot_distribution(x, ax, i, j):
    # B, N, M = x.shape
    # # verts = []
    # if j==0:
    #     length = N
    #     x = x[,,]
    # for t in range(length):
    #     x_total = []
    #     for k in range(30):
    #         x_total.append(x[k][t].cpu().detach().numpy())
    #     # x_average = sum(x_total)/len(x_total)
    #     ys = max(x[0][t].cpu().detach().numpy())
    #     xs = np.arange(M)
    #     # ax.bar(xs, x_average, zs=t, zdir='y', color='g', alpha=0.8)
    #     # verts.append(list(zip(xs, x_average)))
    # ax.plot(xs, ys, color='g', alpha=0.8)
    # # poly = PolyCollection(verts)
    # # poly.set_alpha(0.7)
    # # ax.add_collection3d(poly, zs=t, zdir='y')
    # ax.set_xlabel('Channel')
    # # ax.set_ylabel('Token')
    # # ax.set_zlabel('Value')

    B, M, H, W = x.shape
    # x_max_average = []
    # x_min_average = []
    # if j==0:
    #     length = N
    #     label = 'Token'
    #     x_max = x.max(axis=2).values
    #     x_min = x.min(axis=2).values
    # else:
    length = M
    label = 'Channel'
    x_max = x.max(axis=2).values
    x_min = x.min(axis=2).values
    # x_max_mean = torch.mean(x_max, 0)
    # x_min_mean = torch.mean(x_min, 0)
    x_max_max = x_max.max(axis=2).values
    x_min_min = x_min.min(axis=2).values
    
    x_max_all = x_max_max.max(axis=0).values
    x_min_all = x_min_min.min(axis=0).values
    # for t in range(length):
        # x_max_total = []
        # x_min_total = []
        # for k in range(30):
            # if j==0:
            #     x_new = x.max(axis=2).values
            # else:
            #     x_new = x.max(axis=1).values
            # x_max_total.append(max(x_new))
            # x_min_total.append(min(x_new))
        # x_max_average.append(sum(x_max_total)/len(x_max_total))
        # x_min_average.append(sum(x_min_total)/len(x_min_total))
    xs = np.arange(length)
    ax.plot(xs, x_max_all.cpu().detach().numpy(), color='lightseagreen', alpha=1, label="Max")
    ax.plot(xs, x_min_all.cpu().detach().numpy(), color='royalblue', alpha=0.8, label="Min")
    position = ["lower left", "lower left", "lower right", "upper right"]
    leg = ax.legend(fontsize=7, loc=position[i], ncol=1)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    # ax.set_xlabel(label)

def plot_distribution(a, name, quant=False):
    for i in range(len(a)):
        fig, ax = plt.subplots(1, 1,figsize=(4, 3))
        print("Ploting......")
        # fig = plt.figure(figsize=(10, 25))
        # gs = gridspec.GridSpec(1,4)
        # for i in range(len(a)):
            # for j in range():
        j = 1
        # ax = fig.add_subplot(gs[i], projection='3d')
        sub_plot_distribution(a[i], ax, i, j)
        if quant:
            # TODO:
            name += "_quant"
        # else:
            # name += "_smoothed"
        plt.tight_layout()
        plt.savefig("figs/" + name + "_" + str(i)+ ".svg")


