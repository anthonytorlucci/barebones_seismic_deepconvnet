import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_gather(fig_handle, input_ensemble, output_ensemble):
    # https://stackoverflow.com/questions/23876588/matplotlib-colorbar-in-each-subplot
    # fig = plt.figure(figsize=(16, 12))
    ax1 = fig_handle.add_subplot(131)
    ax1.set_title("input gather")
    im1 = ax1.imshow(numpy.transpose(input_ensemble), clim=(-5000,5000), cmap="RdYlBu", aspect="auto", interpolation='None')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig_handle.colorbar(im1, cax=cax, orientation='vertical')

    ax2 = fig_handle.add_subplot(132)
    ax2.set_title("predicted gather")
    im2 = ax2.imshow(numpy.transpose(output_ensemble), clim=(-5000,5000), cmap="RdYlBu", aspect="auto", interpolation='None')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig_handle.colorbar(im2, cax=cax, orientation='vertical')

    ax3 = fig_handle.add_subplot(133)  # difference, i.e. predicted noise
    ax3.set_title("difference or predicted noise")
    im3 = ax3.imshow(numpy.transpose(input_ensemble - output_ensemble), clim=(-1000,1000), cmap="RdYlBu", aspect="auto", interpolation='None')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig_handle.colorbar(im3, cax=cax, orientation='vertical')