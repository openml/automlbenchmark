import matplotlib as mp
import numpy as np
import seaborn as sb

import amlb_report.config as config
from ..util import create_file, sort_dataframe
from .util import savefig, set_labels, set_scales, task_labels


def draw_barplot(df, x, y, hue=None,
                 x_labels=None, yscale='linear', ylim=None, hue_order=None,
                 title=None, xlabel=None, ylabel=None, ci=None,
                 legend_loc='best', legend_title=None, legend_labels=None,
                 colormap=None, size=None):
    colormap = config.colormap if colormap is None else colormap
    with sb.axes_style('whitegrid', rc={'grid.linestyle': 'dotted'}), sb.plotting_context('paper'):
        bar_fig = mp.pyplot.figure(dpi=120, figsize=size or (len(df[x].unique()), 10))
        # select the first colors from the colormap to ensure we use the same colors as in the stripplot later
        axes = sb.barplot(data=df,
                          x=x, y=y, hue=hue,
                          hue_order=hue_order,
                          ci=ci,
                          palette=colormap)

        set_scales(axes, yscale=yscale)
        if isinstance(ylim, tuple):
            axes.set_ylim(ylim)
        if isinstance(ylim, dict):
            axes.set_ylim(**ylim)

        handles, labels = axes.get_legend_handles_labels()
        if legend_labels is not None:
            if isinstance(legend_labels, list):
                labels = legend_labels
            else:
                labels = map(legend_labels, labels)
        axes.legend(handles, labels, loc=legend_loc, title=legend_title)
        set_labels(axes, title=title, xlabel=xlabel, ylabel=ylabel, x_labels=x_labels,
                   x_tick_params=dict(labelrotation=90))
        return bar_fig


def draw_score_barplot(col, results, type_filter='all', metadata=None,
                       x_sort_by='name', ylabel=None, hue_sort_by=None,
                       filename=None,
                       **kwargs):
    sort_by = (x_sort_by if callable(x_sort_by)
               else None if not metadata or not isinstance(x_sort_by, str)
               else lambda row: row.task.apply(lambda t: getattr(metadata[t], x_sort_by)))
    plot_df = sort_dataframe(results.set_index(['type', 'task']), by=sort_by)
    df = (plot_df if type_filter == 'all'
          else plot_df[plot_df.index.get_loc(type_filter)])
    x_labels = task_labels(df.index.unique())
    df.reset_index(inplace=True)

    hue = 'framework'
    hues = sorted(df[hue].unique(), key=hue_sort_by)

    fig = draw_barplot(df,
                       x='task', y=col, hue=hue,
                       x_labels=x_labels,
                       ylabel=ylabel or "Score",
                       hue_order=hues,
                       legend_title="Framework",
                       **kwargs)
    if filename:
        savefig(fig, create_file("graphics", config.results_group, filename))
    return fig
