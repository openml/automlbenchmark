import seaborn as sb

import report.config as config
from ..util import create_file, sort_dataframe
from .util import savefig, set_labels, task_labels


def draw_heatmap(df,
                 x_labels=True, y_labels=True,
                 title=None, xlabel=None, ylabel=None,
                 **kwargs):
    with sb.axes_style('white'), sb.plotting_context('paper'):
        #         print(sb.axes_style())
        #         print(sb.plotting_context())
        axes = sb.heatmap(df, xticklabels=x_labels, yticklabels=y_labels,
                          annot=True, cmap='RdYlGn', robust=True,
                          **kwargs)
        set_labels(axes, title=title,
                   xlabel=xlabel, ylabel=ylabel,
                   x_tick_params=dict(labelrotation=90))
        fig = axes.get_figure()
        fig.set_size_inches(10, df.shape[0]/2)
        fig.set_dpi(120)
        return fig


def draw_score_heatmap(col, results, type_filter='all', metadata=None, y_sort_by=None,
                       filename=None, **kwargs):
    df = (results.groupby(['type', 'task', 'framework'])[col]
          .mean()
          .unstack())
    df = (df if type_filter == 'all'
          else df[df.index.get_loc(type_filter)])
    if metadata and y_sort_by:
        sort_by = lambda row: row.task.apply(lambda t: getattr(metadata[t], y_sort_by))
        df = sort_dataframe(df, by=sort_by)

    fig = draw_heatmap(df,
                       y_labels=task_labels(df.index),
                       #                        xlabel="Framework", ylabel="Task",
                       **kwargs)
    if filename:
        savefig(fig, create_file("graphics", config.results_group, filename))
    return fig
