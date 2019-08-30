
import matplotlib as mpl


def set_mpl(scale=2):

    scale = scale
    mpl.rcParams['figure.figsize'] = [scale * 3.5, scale * 3.5 / 1.4]
    mpl.rcParams['font.size'] = scale * 11
    mpl.rc('font', family='serif', serif='CMU Serif')
    mpl.rc('xtick', labelsize=scale * 10)
    mpl.rc('ytick', labelsize=scale * 10)
    mpl.rc('axes', labelsize=scale * 11)
    mpl.rc('legend', fontsize=scale * 7)
    mpl.rc('legend', frameon=False)
    mpl.rc('text', usetex=True)
