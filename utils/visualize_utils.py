

"""_utils functions for visualization purposes_
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
from utils.general_utils import cond_mkdir

def plot_loss(iter, loss, filename="loss", label=None, newfig=True, color="b"):
    plt.figure(1)
    plt.clf()
    plt.title(filename)
    plt.xlabel("epoch")
    # plt.ylabel("loss")
    cond_mkdir('imgs/')
    if newfig:
        _ = plt.plot(iter, loss)
        if filename is not None:
            plt.savefig('imgs/'+filename + ".png",
                        dpi=200, bbox_inches="tight")
    plt.draw()
    plt.show()


def show(input, title="image", cut=False, cmap='gray',
         clim=None, rect_list=None, hist=False, save=False,
         save_name='picture', log_scale=False, figure_size=5):

    if log_scale:
        if torch.is_tensor(input):
            input = torch.log(input)
        else:
            input = np.log(input)

    if torch.is_tensor(input):
        if input.device.type != 'cpu':
            print('detect the cuda')
            input = input.cpu()

    if hist:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121)
    else:
        fig = plt.figure(figsize=(figure_size, figure_size))
        ax = fig.add_subplot(111)

    ax.title.set_text(title)
    if cut:
        img = ax.imshow(input, cmap=cmap, vmin=0, vmax=1)
    else:
        img = ax.imshow(input, cmap=cmap)
    plt.colorbar(img, ax=ax)

    if rect_list is not None:
        # rect_params contain [(x, y), width, height]
        rect = patches.Rectangle(
            rect_list[0], rect_list[1], rect_list[2],
            linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    if hist:
        ax_ = fig.add_subplot(122)
        n, bins, patches = ax_.hist(input.flatten(), 100)
        ax_.title.set_text(title)

    if save is True:
        plt.savefig(save_name + '.png')
    plt.show()
