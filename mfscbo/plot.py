import matplotlib.pyplot as plt
import numpy as np
from mfscbo.utils import COLORS, MARKERS, ensure_list


def get_alphas(num_fidelities):
    """Get alpha values for plotting based on the number of fidelities.

    Args:
        num_fidelities (int): Number of fidelities.
    Returns:
        list of float: List of alpha values for each fidelity.
    """
    alphas = [(i+1)/num_fidelities for i in range(num_fidelities)]
    return alphas

def plot_rhos(rhos, 
              xvalues, 
              ax=None, 
              path=None,
              labels=None,
              markers=None,
              colors=None,
              alphas=None,
              linewidth=None,
              markersize=None,
              markevery=None
              ):
    """Plot the evolution of the rho parameters over iterations.

    Args:
        rhos (List[List[float]]): List containing the rho values for each fidelity at each iteration.
        xvalues (List[float]): List of x values.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, a new figure and axes are created.
        path (str, optional): Path to save the plot. If None, the plot is not saved.
        labels (list or str, optional): List of labels for each fidelity. If None, default labels are used.
        markers (list or str, optional): List of markers for each fidelity. If None, default markers are used.
        colors (list or str, optional): List of colors for each fidelity. If None, default colors are used.
        alphas (list or float, optional): List of alpha values for each fidelity. If None, default alphas are used.
        linewidth (list or float, optional): List of linewidths for each fidelity. If None, default linewidths are used.
        markersize (list or float, optional): List of markersizes for each fidelity. If None, default markersizes are used.
        markevery (list or float, optional): List of markevery values for each fidelity. If None, default markevery values are used.

    Returns:
        matplotlib.axes.Axes: The axes object with the plot.
    """

    is_new_figure = False
    if ax is None:
        is_new_figure = True
        fig, ax = plt.subplots(figsize=(8, 6))
       
    num_fidelities = len(rhos)#do not consider fidelity 0
    labels = ensure_list(labels, [f's={i+1}' for i in range(num_fidelities)], num_fidelities)
    markers = ensure_list(markers, [MARKERS[i+1] for i in range(num_fidelities)], num_fidelities)
    colors = ensure_list(colors, [COLORS[3]] * num_fidelities, num_fidelities)
    alphas = ensure_list(alphas, get_alphas(num_fidelities), num_fidelities)
    linewidth = ensure_list(linewidth, [2] * num_fidelities, num_fidelities)
    markersize = ensure_list(markersize, [5] * num_fidelities, num_fidelities)
    markevery = ensure_list(markevery, [0.2] * num_fidelities, num_fidelities)


    for i in range(num_fidelities):
        ax.plot(xvalues, rhos[i], marker=markers[i], label=f's={i+1}', color=colors[i], alpha=alphas[i], linewidth=linewidth[i], markersize=markersize[i], markevery=markevery[i])

    if is_new_figure:
        ax.set_ylabel(r'$\rho$')
        ax.set_title(r'Evolution of $\rho$')
        ax.legend()
        ax.grid()
        plt.show()

    if path is not None:
        plt.savefig(path)
    
    return ax

def plot_noises(noises, 
                xvalues, 
                ax=None, 
                path=None,
                labels=None,
                markers=None,
                colors=None,
                alphas=None,
                linewidth=None,
                markersize=None,
                markevery=None) :
    """Plot the evolution of the noise parameters over iterations.

    Args:
        noises (List[List[float]]): List containing the noise values for each fidelity at each iteration.
        xvalues (List[float]): List of x values.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, a new figure and axes are created.
        path (str, optional): Path to save the plot. If None, the plot is not saved.
        labels (list or str, optional): List of labels for each fidelity. If None, default labels are used.
        markers (list or str, optional): List of markers for each fidelity. If None, default markers are used.
        colors (list or str, optional): List of colors for each fidelity. If None, default colors are used.
        alphas (list or float, optional): List of alpha values for each fidelity. If None, default alphas are used.
        linewidth (list or float, optional): List of linewidths for each fidelity. If None, default linewidths are used.
        markersize (list or float, optional): List of markersizes for each fidelity. If None, default markersizes are used.
        markevery (list or float, optional): List of markevery values for each fidelity. If None, default markevery values are used.

    Returns:
        matplotlib.axes.Axes: The axes object with the plot.
    """

    is_new_figure = False
    if ax is None:
        is_new_figure = True
        fig, ax = plt.subplots(figsize=(8, 6))

    num_fidelities = len(noises) #consider fidelity 0
    labels = ensure_list(labels, [f's={i}' for i in range(num_fidelities)], num_fidelities)
    markers = ensure_list(markers, [MARKERS[i] for i in range(num_fidelities)], num_fidelities)
    colors = ensure_list(colors, [COLORS[2]] * num_fidelities, num_fidelities)
    alphas = ensure_list(alphas, get_alphas(num_fidelities), num_fidelities)
    linewidth = ensure_list(linewidth, [2] * num_fidelities, num_fidelities)
    markersize = ensure_list(markersize, [5] * num_fidelities, num_fidelities)
    markevery = ensure_list(markevery, [0.2] * num_fidelities, num_fidelities)

    for i in range(num_fidelities):
        ax.plot(xvalues, noises[i], marker=markers[i], label=labels[i], color=colors[i], alpha=alphas[i], linewidth=linewidth[i], markersize=markersize[i], markevery=markevery[i])

    if is_new_figure:
        ax.set_ylabel(r'$\varepsilon$')
        ax.set_title(r'Evolution of $\varepsilon$')
        ax.set_yscale('log')    
        ax.legend()
        ax.grid()
        plt.show()

    if path is not None:
        plt.savefig(path)

    return ax

def plot_bestYs(bestYs, 
                xvalues, 
                ax=None, 
                path=None,
                label=None,
                marker=None,
                color=None,
                alpha=None,
                linewidth=None,
                markersize=None,
                markevery=None) :
    """Plot the evolution of the best objective values over iterations.

    Args:
        bestYs (List[float]): List containing the best objective values at each iteration.
        xvalues (List[float]): List of x values.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, a new figure and axes are created.
        path (str, optional): Path to save the plot. If None, the plot is not saved.
        label (str, optional): Label for the plot. If None, a default label is used.
        marker (str, optional): Marker style for the plot. If None, a default marker is used.
        color (str, optional): Color for the plot. If None, a default color is used.
        alpha (float, optional): Alpha value for the plot. If None, a default alpha is used.
        linewidth (float, optional): Linewidth for the plot. If None, a default linewidth is used.
        markersize (float, optional): Markersize for the plot. If None, a default markersize is used.
        markevery (float, optional): Markevery value for the plot. If None, a default markevery value is used.

    Returns:
        matplotlib.axes.Axes: The axes object with the plot.
    """

    is_new_figure = False
    if ax is None:
        is_new_figure = True
        fig, ax = plt.subplots(figsize=(8, 6))

    if color is None:
        color = COLORS[1]
    if linewidth is None:
        linewidth = 2

    ax.plot(xvalues, bestYs, color=color, linewidth=linewidth, label=label, marker=marker, alpha=alpha, markersize=markersize, markevery=markevery)

    if is_new_figure:
        ax.set_title('Evolution of Best Objective Value')
        ax.legend()
        ax.grid()
        plt.show()

    if path is not None:
        plt.savefig(path)

    return ax

def plot_nbiters(nbiters, 
                 xvalues, 
                 ax=None, 
                 path=None,
                 labels=None,
                 markers=None,
                 colors=None,
                 alphas=None,
                 linewidth=None,
                 markersize=None,
                 markevery=None) :
    """Plot the number of iterations at each fidelity over total cost.

    Args:
        nbiters (List[List[int]]): List containing the number of iterations for each fidelity at each total cost.
        xvalues (List[float]): List of x values (total costs).
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, a new figure and axes are created.
        path (str, optional): Path to save the plot. If None, the plot is not saved.
        labels (list or str, optional): List of labels for each fidelity. If None, default labels are used.
        markers (list or str, optional): List of markers for each fidelity. If None, default markers are used.
        colors (list or str, optional): List of colors for each fidelity. If None, default colors are used.
        alphas (list or float, optional): List of alpha values for each fidelity. If None, default alphas are used.
        linewidth (list or float, optional): List of linewidths for each fidelity. If None, default linewidths are used.
        markersize (list or float, optional): List of markersizes for each fidelity. If None, default markersizes are used.
        markevery (list or float, optional): List of markevery values for each fidelity. If None, default markevery values are used.

    Returns:
        matplotlib.axes.Axes: The axes object with the plot.
    """

    is_new_figure = False
    if ax is None:
        is_new_figure = True
        fig, ax = plt.subplots(figsize=(8, 6))

    num_fidelities = len(nbiters)
    labels = ensure_list(labels, [f's={i}' for i in range(num_fidelities)], num_fidelities)
    markers = ensure_list(markers, [MARKERS[i] for i in range(num_fidelities)], num_fidelities)
    colors = ensure_list(colors, [COLORS[0]] * num_fidelities, num_fidelities)
    alphas = ensure_list(alphas, get_alphas(num_fidelities), num_fidelities)
    linewidth = ensure_list(linewidth, [2] * num_fidelities, num_fidelities)
    markersize = ensure_list(markersize, [5] * num_fidelities, num_fidelities)
    markevery = ensure_list(markevery, [0.2] * num_fidelities, num_fidelities)

    for i in range(num_fidelities):
        ax.plot(xvalues, nbiters[i], marker=markers[i], label=labels[i], color=colors[i], alpha=alphas[i], linewidth=linewidth[i], markersize=markersize[i], markevery=markevery[i])

    if is_new_figure:
        ax.set_title('Number of Iterations per Fidelity')
        ax.legend()
        ax.grid()
        plt.show()

    if path is not None:
        plt.savefig(path)

    return ax


def plot_tr_lengths(tr_lengths, 
                   xvalues, 
                   ax=None, 
                   path=None,
                   label=None,
                   marker=None,
                   color=None,
                   alpha=None,
                   linewidth=None,
                   markersize=None,
                   markevery=None) :
    """Plot the evolution of the trust-region lengths over iterations.

    Args:
        tr_lengths (List[float]): List containing the trust-region lengths at each iteration.
        xvalues (List[float]): List of x values.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, a new figure and axes are created.
        path (str, optional): Path to save the plot. If None, the plot is not saved.
        label (str, optional): Label for the plot. If None, a default label is used.
        marker (str, optional): Marker style for the plot. If None, a default marker is used.
        color (str, optional): Color for the plot. If None, a default color is used.
        alpha (float, optional): Alpha value for the plot. If None, a default alpha is used.
        linewidth (float, optional): Linewidth for the plot. If None, a default linewidth is used.
        markersize (float, optional): Markersize for the plot. If None, a default markersize is used.
        markevery (float, optional): Markevery value for the plot. If None, a default markevery value is used.

    Returns:
        matplotlib.axes.Axes: The axes object with the plot.
    """

    is_new_figure = False
    if ax is None:
        is_new_figure = True
        fig, ax = plt.subplots(figsize=(8, 6))

    if color is None:
        color = "k"
    if linewidth is None:
        linewidth = 2

    ax.plot(xvalues, tr_lengths, color=color, linewidth=linewidth, label=label, marker=marker, alpha=alpha, markersize=markersize, markevery=markevery)

    if is_new_figure:
        ax.set_title('Evolution of Trust-Region Lengths')
        ax.legend()
        ax.grid()
        plt.show()

    if path is not None:
        plt.savefig(path)

    return ax

