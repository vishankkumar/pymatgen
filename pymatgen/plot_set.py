
import matplotlib as mpl


def set_mpl(width=6, labelsize=None, ticksize=None):

    width = width
    ticksize = int(width * 2)
    labelsize = int(width * 2.2)

    mpl.rcParams['figure.figsize'] = [width, width / 1.4]
    mpl.rcParams['font.size'] = labelsize #22
    # mpl.rcParams['lines.linewidth'] = width * 1
    # mpl.rcParams['lines.markersize'] = 6

    mpl.rc('font', family='serif', serif='DejaVu Sans')
    mpl.rc('axes', labelsize=labelsize)
    mpl.rc('legend', frameon=False)
    mpl.rc('text', usetex=True)
    mpl.rc('legend', fontsize=ticksize)
    mpl.rc('xtick', labelsize=ticksize)
    mpl.rc('ytick', labelsize=ticksize)
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 400
