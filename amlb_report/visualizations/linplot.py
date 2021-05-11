import matplotlib as mp
import pandas as pd
import seaborn as sb

import amlb_report.config as config
from ..util import create_file, sort_dataframe
from .util import savefig, set_scales, set_labels, task_labels


def draw_parallel_coord(df, class_column,
                        x_labels=True, yscale='linear',
                        title=None, xlabel=None, ylabel=None,
                        legend_loc='best', legend_title=None,
                        colormap=None, size=None):
    colormap = config.colormap if colormap is None else colormap
    with sb.axes_style('ticks', rc={'grid.linestyle': 'dotted'}), sb.plotting_context('paper'):
        #         print(sb.axes_style())
        parallel_fig = mp.pyplot.figure(dpi=120, figsize=size or (10, df.shape[0]))
        # select the first colors from the colormap to ensure we use the same colors as in the stripplot later
        colors = mp.cm.get_cmap(colormap).colors[:len(df[class_column].unique())]
        axes = pd.plotting.parallel_coordinates(df,
                                                class_column=class_column,
                                                color=colors,
                                                axvlines=False,
                                                )
        set_scales(axes, yscale=yscale)
        handles, labels = axes.get_legend_handles_labels()
        axes.legend(handles, labels, loc=legend_loc, title=legend_title)
        set_labels(axes, title=title, xlabel=xlabel, ylabel=ylabel, x_labels=x_labels,
                   x_tick_params=dict(labelrotation=90))
        return parallel_fig


def draw_score_parallel_coord(col, results, type_filter='all', metadata=None,
                              x_sort_by='name', ylabel=None, filename=None,
                              **kwargs):
    res_group = results.groupby(['type', 'task', 'framework'])
    df = res_group[col].mean().unstack(['type', 'task'])
    df = (df if type_filter == 'all'
          else df.iloc[:, df.columns.get_loc(type_filter)])
    sort_by = (x_sort_by if callable(x_sort_by)
               else None if not metadata or not isinstance(x_sort_by, str)
               else lambda cols: getattr(metadata[cols[1]], x_sort_by))
    df = sort_dataframe(df, by=sort_by, axis=1)
    df.reset_index(inplace=True)
    fig = draw_parallel_coord(df,
                              'framework',
                              x_labels=task_labels(df.columns.drop('framework')),
                              # xlabel="Task",
                              ylabel=ylabel or "Score",
                              legend_title="Framework",
                              **kwargs)
    if filename:
        savefig(fig, create_file("graphics", config.results_group, filename))
    return fig
