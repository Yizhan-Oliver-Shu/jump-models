"""
module for plotting
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from .utils import *

ALPHA_LINE = .8
ALPHA_FILL = .3
AXES_TYPE = Optional[plt.Axes]

############################
## matplotlib setting
############################

# checked
def matplotlib_setting():
    """
    set rcParams globally.
    """
    plt.rcParams['figure.figsize'] = (24, 12)
    plt.rcParams['axes.titlesize'] = 30
    plt.rcParams['axes.labelsize'] = 30
    plt.rcParams['xtick.labelsize'] = 30
    plt.rcParams['ytick.labelsize'] = 30
    plt.rcParams['legend.fontsize'] = 30
    plt.rcParams['font.size'] = 26
    plt.rcParams['font.family'] = 'cmr10'
    plt.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    return 

# set params for plotting
matplotlib_setting()

############################
## i/o
############################

# checked
def check_axes(ax: AXES_TYPE = None, nrows=1, ncols=1, figsize_single=(24, 12), **kwargs) -> Union[plt.Axes, np.ndarray]:
    """
    if `ax` is None, create one. else return `ax`.
    """
    if ax is None:
        w, h = figsize_single
        _, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*w, nrows*h), **kwargs)
    # assert isinstance(ax, plt.Axes)
    return ax

# checked
def savefig_plt(filepath, close=True):
    """
    save figure to a path. will automatically create the folder if non-existent.
    """
    check_dir_exist(filepath)
    plt.savefig(filepath)
    if close: plt.close()
    return 

# checked
def make_twin_ax(ax: plt.Axes) -> plt.Axes:
    """
    return the `twinx` of the input axes
    """
    return ax.twinx()

############################
## plot horizontal line
############################

# checked
def plot_hline(ax: AXES_TYPE = None, yvalue: float = 0., color='r', linestyle='--') -> plt.Axes:
    ax = check_axes(ax)
    ax.axhline(yvalue, color=color, linestyle=linestyle)
    return ax

############################
## plot cum ret
############################

# y-axis
from matplotlib.ticker import FuncFormatter, FormatStrFormatter

# checked
def convert_yaxis_to_percent(ax: plt.Axes, num_digits=0) -> None:
    """
    convert the ticks on the y axis to percent
    """
    # convert to percent
    def to_percent(x, position):
        return float_to_percent_str(x, num_digits=num_digits, latex=True)
    ax.yaxis.set_major_formatter(FuncFormatter(to_percent)) 
    return 

# # checked
# def convert_yaxis_to_float_format(ax: plt.Axes, num_digits: int = 1) -> None:
#     """
#     convert the ticks on the y axis to float number with a specific number of digits.
#     """
#     format = r"%." + str(num_digits) + "f"
#     ax.yaxis.set_major_formatter(FormatStrFormatter(format))
#     return 

# checked
def plot_cumret(ret_df: Union[PD_TYPE, dict], 
                start_date: DATE_TYPE = None, 
                end_date: DATE_TYPE = None, 
                ax: AXES_TYPE = None, 
                ytick_percent=True, 
                ytick_num_digits=0,
                ylabel="Cumulative Excess Returns",
                hline=True,
                legend=True,
                loc=None
                ) -> plt.Axes:
    """
    compute and plot the cumulative return from a ret df.
    """
    ax = check_axes(ax)
    # process `ret_df`
    ret_df = filter_date_range(pd.DataFrame(ret_df), start_date, end_date)
    ret_df.index.name = None
    # plot cumret
    ret_df.cumsum(axis=0).plot(ax=ax, legend=legend)
    # set ax attrs
    ax.set(ylabel=ylabel)
    if ytick_percent: convert_yaxis_to_percent(ax, num_digits=ytick_num_digits)
    if hline: plot_hline(ax)
    if legend: ax.legend(loc=loc)
    return ax

# ############################
# ## basic calculation
# ############################

# # correct
# def compute_crash_stats_2_regimes(labels_: SER_ARR_TYPE) -> tuple[float, int]:
#     """
#     compute the percent of crashes and the number of regime shifts, from a label sequence.
#     only work for 2-regime
#     """
#     n_c = 2
#     assert is_valid_labels(labels_, n_c=n_c)
#     return labels_.mean(), compute_num_shifts(labels_)

# # correct
# def generate_crash_stats_str_2_regime(labels_: SER_ARR_TYPE) -> str:
#     """
#     returns a stats str
#     """
#     percent_crash, num_shifts = compute_crash_stats_2_regimes(labels_)
#     return r"\% of Bear Market: " + float_to_percent_str(percent_crash, num_digits=1) + r", Number of Regime Shifts: " + f"{num_shifts}"

############################
## plot regimes
############################

# checked
def ax_fill_between(labels_: pd.Series,
                    start_date: DATE_TYPE = None, 
                    end_date: DATE_TYPE = None, 
                    ax: AXES_TYPE = None, 
                    color: Optional[str] = None, 
                    fill_between_label: Optional[str] = None) -> plt.Axes:
    """
    fill the color for a single labels series
    call the `fill_between` method of an axes instance.
    """
    ax = check_axes(ax)
    # filter dates
    labels_ = filter_date_range(labels_, start_date, end_date)
    # plot
    ax.fill_between(labels_.index, labels_, step="pre", alpha=ALPHA_FILL, color=color, label=fill_between_label)

# checked
LABELS_DICT = dict(zip(range(3), ["Bull", "Neutral", "Bear"])) 
COLOR_DICT = dict(zip(range(3), list("gyr"))) 

def plot_regimes(labels_: pd.Series,
                 n_c: int = 3,
                 start_date: DATE_TYPE = None, 
                 end_date: DATE_TYPE = None, 
                 ax: AXES_TYPE = None, 
                 legend=True,
                 loc="upper left"
                 ) -> plt.Axes:
    """
    plot the regimes for 2- and 3-regime labels.
    """
    labels_ = labels_.copy()
    # check inputs
    ax = check_axes(ax)
    assert n_c in [2, 3]
    assert is_valid_labels(labels_, n_c=n_c)
    # trick by unifying 2- and 3-regime treatments
    if n_c == 2: labels_ *= 2.
    for i in range(3):
        if i == 1 and n_c == 2: continue
        ax_fill_between(labels_.eq(i)*1., start_date=start_date, end_date=end_date, ax=ax, color=COLOR_DICT[i], fill_between_label=LABELS_DICT[i])
    if legend: ax.legend(loc=loc)
    ax.set(yticks=[])
    return ax

# checked
def plot_cumret_regimes(ret_df: Union[dict, pd.DataFrame],
                        labels_: pd.Series,
                        start_date: DATE_TYPE = None, 
                        end_date: DATE_TYPE = None, 
                        n_c: int = 3,
                        ax: AXES_TYPE = None, 
                        ytick_percent_ret=True, 
                        ytick_num_digits_ret=0,
                        ylabel_ret="Cumulative Excess Return",
                        hline_ret=True,
                        loc=None
                        ) -> tuple[plt.Axes, plt.Axes]:
    if start_date is None and end_date is None: start_date, end_date = labels_.index[[0, -1]]
    ax = check_axes(ax)

    # plot regimes
    ax2 = ax.twinx()
    plot_regimes(labels_, n_c=n_c, start_date=start_date, end_date=end_date, ax=ax2, legend=False)

    # plot cumret
    plot_cumret(ret_df, start_date=start_date, end_date=end_date, ax=ax, ytick_percent=ytick_percent_ret, ytick_num_digits=ytick_num_digits_ret, ylabel=ylabel_ret, hline=hline_ret, legend=False)

    # Merge the labels and handles from both axes
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax2.legend(lines + lines2, labels + labels2, loc=loc)
    return (ax, ax2)


# # correct
# def plot_regime(labels_: pd.Series, 
#                 start_date=None,
#                 end_date=None,
#                 ax: plt.Axes = None, 
#                 title: str = None, 
#                 stats_on_title: bool = True, 
#                 color: str = "r", 
#                 legend: bool = True,
#                 fill_between_label: str = "Bear") -> plt.Axes:
#     """
#     plot the regimes. 

#     Parameters:
#     ------------------
#     regimes: ser/df
#     """
#     assert is_ser(labels_)
#     labels_ = filter_date_range(labels_, start_date, end_date)
#     assert is_valid_labels(labels_, n_c=2)
#     ax = check_axes(ax)
#     ax.set(yticks=[])
#     # plot regimes
#     ax.fill_between(labels_.index, labels_, step="pre", alpha=ALPHA_FILL, color=color, label=fill_between_label)
#     # title
#     if stats_on_title:
#         title = check_string(title) + ("" if title is None else " ") 
#         title += generate_crash_stats_str_2_regime(labels_)
#     if title is not None: ax.set_title(title)
#     if legend: ax.legend(loc='center left')    # loc='upper left'
#     return ax



# # correct
# def plot_regime(labels_: pd.Series, 
#                 start_date=None,
#                 end_date=None,
#                 ax: plt.Axes = None, 
#                 title: str = None, 
#                 stats_on_title: bool = True, 
#                 color: str = "r", 
#                 legend: bool = True,
#                 fill_between_label: str = "Bear") -> plt.Axes:
#     """
#     plot the regimes. 

#     Parameters:
#     ------------------
#     regimes: ser/df
#     """
#     assert is_ser(labels_)
#     labels_ = filter_date_range(labels_, start_date, end_date)
#     assert is_valid_labels(labels_, n_c=2)
#     ax = check_axes(ax)
#     ax.set(yticks=[])
#     # plot regimes
#     ax.fill_between(labels_.index, labels_, step="pre", alpha=ALPHA_FILL, color=color, label=fill_between_label)
#     # title
#     if stats_on_title:
#         title = check_string(title) + ("" if title is None else " ") 
#         title += generate_crash_stats_str_2_regime(labels_)
#     if title is not None: ax.set_title(title)
#     if legend: ax.legend(loc='center left')    # loc='upper left'
#     return ax

# # correct
# def plot_regime_and_cumret(labels_: pd.Series, ret_df, start_date=None, end_date=None, ax=None, title=None, stats_on_title=True, fill_between_label: str = "Bear", special_color: bool = False) -> tuple[plt.Axes, plt.Axes]:
#     # valid inputs
#     ret_df = pd.DataFrame(ret_df)
#     labels_ = filter_date_range(labels_, start_date, end_date)
#     ret_df = align_x_with_y(ret_df, labels_)
#     # assert is_same_index(labels_, ret_df)
#     # check axes
#     ax = check_axes(ax)
#     ax2 = ax.twinx()
#     # plot cumret
#     plot_cumret(ret_df=ret_df, ax=ax, special_color=special_color)
#     # plot regime
#     plot_regime(labels_=labels_, ax=ax2, title=title, stats_on_title=stats_on_title, fill_between_label=fill_between_label)
#     return ax, ax2

# ############################
# ## plot price df
# ############################

# # reviewed
# YTICKS_WEALTH = [.1, .2, .5, .8, 1., 2., 5., 10., 20., 50., 100.]
# def set_yaxis_logscale(ax=None, adjust_ticks=YTICKS_WEALTH):
#     """
#     set the scale of y-axis to be log, with specific ticks to show.
#     """
#     ax = check_axes(ax)
#     assert isinstance(ax, plt.Axes)
#     ax.set_yscale('log')
#     if adjust_ticks:
#         ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
#         ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
#         b, t = ax.get_ylim()
#         ax.set(yticks=[y for y in adjust_ticks if y>=b and y<=t])
#     return ax

