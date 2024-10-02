"""
module for plotting
"""

from .utils import *

import matplotlib.pyplot as plt
import matplotlib as mpl

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