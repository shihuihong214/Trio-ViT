import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.collections import PolyCollection
import torch
import numpy as np

def sub_plot_distribution(x, ax, i, j):

    B, M, H, W = x.shape
    if j == 0:
        length = M
        label = 'Filter'
        x_max = x.max(axis=2).values
        x_min = x.min(axis=2).values
        # x_max_mean = torch.mean(x_max, 0)
        # x_min_mean = torch.mean(x_min, 0)
        x_max_max = x_max.max(axis=2).values
        x_min_min = x_min.min(axis=2).values
        
        x_max_all = x_max_max.max(axis=0).values
        x_min_all = x_min_min.min(axis=0).values
    
    elif j == 1:
        length = H
        label = 'Row'
        x_max = x.max(axis=3).values
        x_min = x.min(axis=3).values
        # x_max_mean = torch.mean(x_max, 0)
        # x_min_mean = torch.mean(x_min, 0)
        x_max_max = x_max.max(axis=1).values
        x_min_min = x_min.min(axis=1).values
        
        x_max_all = x_max_max.max(axis=0).values
        x_min_all = x_min_min.min(axis=0).values
    
    elif j == 2:
        length = W
        label = 'Column'
        x_max = x.max(axis=2).values
        x_min = x.min(axis=2).values
        # x_max_mean = torch.mean(x_max, 0)
        # x_min_mean = torch.mean(x_min, 0)
        x_max_max = x_max.max(axis=1).values
        x_min_min = x_min.min(axis=1).values
        
        x_max_all = x_max_max.max(axis=0).values
        x_min_all = x_min_min.min(axis=0).values
    percentile_alpha = 0.90
    try:
        cur_max = torch.quantile(x.reshape(-1), percentile_alpha)
    except:
        cur_max = torch.tensor(np.percentile(
            x.reshape(-1).cpu().detach().numpy(), percentile_alpha * 100),
                                device=x.device,
                                dtype=torch.float32)
    print(x_max_all.max()/cur_max)

    print(x_max_all.max(), x_min_all.min())
    print(x_max_all[0], x_min_all[0])
    xs = np.arange(length)
    ax.plot(xs, x_max_all.cpu().detach().numpy(), color='lightseagreen', alpha=1, label="Max")
    ax.plot(xs, x_min_all.cpu().detach().numpy(), color='royalblue', alpha=0.8, label="Min")
    position = ["lower left", "upper left", "lower right", "upper right"]
    leg = ax.legend(fontsize=9, loc=position[3], ncol=1)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    # ax.set_xlabel(label)
   

def LogQuant(x):
    y = torch.floor(torch.div(torch.log(x),torch.log(torch.Tensor([2]))))
    out = torch.gt(x/(2**y),2**(y+1)/x)
    y += out
    # TODO:
    y = torch.clamp(y, -6, 10)
    # return 2**y
    out = 2**y
    # out[x==0] = 0
    return out
    
def sub_plot_hint(x, ax):
    ax.hist(x.flatten().cpu().detach().numpy(), bins=300, edgecolor='royalblue', alpha=0.8)
    position = ["lower left", "lower left", "lower right", "upper right"]
    # leg = ax.legend(fontsize=7, loc=position[i], ncol=1)
    # leg.get_frame().set_edgecolor("black")
    # leg.get_frame().set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    # ax2 = ax.twinx()
    # x_ax=torch.arange(x.min().detach(), x.max().detach(), step=0.05).cpu()
    # bit = 4
    # delta = ((x.max()-x.min())/(2**bit-1)).cpu()
    # zero_point = (-x.min() / delta).round().cpu()
    # y1 = (x_ax / delta).round() + zero_point
    # y1 = torch.clamp(y1, 0, 2**bit - 1)
    # y1 = (y1 - zero_point) * delta
    # ax2.plot(x_ax.tolist(), y1.tolist(), label="Uniform", color='green')
    # # print(LogQuant(x_ax))
    # ax2.plot(x_ax.tolist(), LogQuant(x_ax), label="Log2", color='orange')
    # position = ["lower left", "upper left", "lower right", "upper right"]
    # lines_1, labels_1 = ax.get_legend_handles_labels()
    # lines_2, labels_2 = ax2.get_legend_handles_labels()
    # lines = lines_1 + lines_2
    # labels = labels_1 + labels_2
    # leg = ax.legend(lines, labels, loc='lower right', fontsize=9)
    # # leg = ax.legend(fontsize=9, loc=position[2], ncol=1)
    # leg.get_frame().set_edgecolor("black")
    # leg.get_frame().set_linewidth(1.5)
    

def plot_distribution(a, name, quant=False):
    for i in range(len(a)):
        print("Ploting......")
        # # fig = plt.figure(figsize=(10, 25))
        # # gs = gridspec.GridSpec(1,4)
        # # for i in range(len(a)):
        # for j in range(3):
        #     fig, ax = plt.subplots(1, 1,figsize=(4, 3))
        #     sub_plot_distribution(a[i], ax, i, j)
        #     if quant:
        #         # TODO:
        #         name += "_quant"
        #     # else:
        #         # name += "_smoothed"
        #     plt.tight_layout()
        #     plt.savefig("figs/" + name + "_" + str(j)+ ".svg")
        
        fig, ax = plt.subplots(1, 1,figsize=(4.3, 3))
        sub_plot_hint(a[i], ax,)
        plt.tight_layout()
        plt.savefig("figs/" + name + "_distr" + ".svg")
        
        # fig, ax = plt.subplots(1, 1,figsize=(4, 3))
        # a = torch.tensor(0)
        # b = torch.tensor(5)
        # c = torch.tensor(1600)
        # x_ax=torch.arange(a, b, step=0.01)
        # bit = 8
        # delta = ((c-a)/(2**bit-1))
        # zero_point = (-a / delta).round()
        # y1 = (x_ax / delta).round() + zero_point
        # y1 = torch.clamp(y1, 0, 2**bit - 1)
        # y1 = (y1 - zero_point) * delta
        # ax.plot(x_ax.tolist(), y1.tolist(), label="Uniform", color='green')
        # # print(LogQuant(x_ax))
        # ax.plot(x_ax.tolist(), LogQuant(x_ax), label="Log2", color='orange')
        # position = ["lower left", "upper left", "lower right", "upper right"]
        # leg = ax.legend(loc='lower right', fontsize=9)
        # # leg = ax.legend(fontsize=9, loc=position[2], ncol=1)
        # leg.get_frame().set_edgecolor("black")
        # leg.get_frame().set_linewidth(1.5)
        # plt.tight_layout()
        # plt.savefig("figs/" + name + "_distr_local" + ".svg")


def plot_MB_distribution(a, name, quant=False):
    
    print("Ploting......")
        # fig = plt.figure(figsize=(10, 25))
        # gs = gridspec.GridSpec(1,4)
        # for i in range(len(a)):
    label = ['C', 'H', 'W']   
    for i in range(1):    
        fig, ax = plt.subplots(1, 1,figsize=(4, 3))
        sub_plot_distribution(a[0], ax, 0, i)
        if quant:
            # TODO:
            name += "_quant"
        # else:
            # name += "_smoothed"
        plt.tight_layout()
        plt.savefig("figs/" + name + "_" + label[i] + ".svg")
     
    fig, ax = plt.subplots(1, 1,figsize=(4, 3))
    sub_plot_hint(a[0], ax)
    plt.tight_layout()
    plt.savefig("figs/" + name + "2_distr" + ".svg")

