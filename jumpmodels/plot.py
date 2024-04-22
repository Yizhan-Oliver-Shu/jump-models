"""
module for plotting
"""

import pandas as pd
import matplotlib.pyplot as plt

from .utils import *

# reviewed
def check_axes(ax=None, nrows=1, ncols=1, figsize_single=(24, 12), **kwargs) -> Union[plt.Axes, np.ndarray]:
    """
    if ax is None, create one.
    """
    if ax is None:
        w, h = figsize_single
        _, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*w, nrows*h), **kwargs)
    return ax

############################
## basic calculation
############################

# reviewed
def compute_crash_stats_2_regimes(labels_):
    """
    compute the percent of crashes and the number of regime shifts, from a label sequence.
    only work for 2-regime
    """
    n_c = 2
    assert is_valid_labels(labels_, n_c=n_c)
    return labels_.mean(), compute_num_shifts(labels_)

# reviewed
def generate_crash_stats_str_2_regime(labels_):
    """
    returns a stats str
    """
    percent_crash, num_shifts = compute_crash_stats_2_regimes(labels_)
    return r"% of crash: " + f"{percent_crash*100:.1f}" + r"%, # of regime shifts: " + f"{num_shifts}"

############################
## plot regimes
############################

def plot_regime(regimes: PD_TYPE, 
                ax: plt.Axes = None, 
                title: str = None, 
                stats_on_title: bool = True, 
                color: str = "r", 
                legend: bool = True,
                fill_between_label: str = "regime"):
    """
    plot the regimes. 
    """
    assert is_ser_df(regimes)
    ax = check_axes(ax)
    # process multi-regime label
    if regimes.ndim == 1 and regimes.max() > 1: 
        n_c = int(regimes.max()) + 1    # should be an int
        assert is_valid_labels(regimes, n_c)
        regimes = pd.DataFrame(raise_labels_into_proba(regimes, n_c), index=regimes.index)  # raise to proba mx
    proba_input = (regimes.ndim == 2)
    if proba_input:
        n_c = regimes.shape[1]
        assert n_c in [2, 3, 4]
        freq = normalize_weight(regimes.sum(axis=0)).to_numpy()
        # if is_ser_df(freq): freq = freq.to_numpy()
        COLOR_LIST = {
            2: ["r"],
            3: ["g", "r"],
            4: ["g", "y", "r"]
        }[n_c]
        for i in range(1, n_c):
            plot_regime(regimes.iloc[:, i], ax=ax, stats_on_title=False, color=COLOR_LIST[i-1], legend=False, fill_between_label=f"regime {i} ({int(np.round(freq[i]*100))}" + r"%)")
    else:
        regime_arr = check_1d_array(regimes)
        index = regimes.index
        # plot regimes
        ax.fill_between(index, regime_arr, step="pre", alpha=.3, color=color, label=fill_between_label)
        ax.set(ylabel='regime', yticks = [0, 1])
    # title
    if stats_on_title:
        title = check_string(title) + ("" if title is None else ", ") 
        if proba_input:
            labels = reduce_proba_to_labels(regimes)
            title += r"# of regime shifts: " + f"{compute_num_shifts(labels)}"
        else:
            title += generate_crash_stats_str_2_regime(regimes)
    if title is not None: ax.set_title(title)
    if legend: ax.legend(loc='center left')
    return ax
