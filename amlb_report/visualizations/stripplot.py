import matplotlib as mp
import seaborn as sb

import amlb_report.config as config
from ..util import create_file, sort_dataframe
from .util import savefig, set_scales, set_labels, task_labels


def draw_stripplot(df, x, y, hue,
                   xscale='linear', xbound=None, hue_order=None,
                   xlabel=None, ylabel=None, y_labels=None, title=None,
                   legend_title=None, legend_loc='best', legend_labels=None,
                   colormap=None, size=None):
    colormap = config.colormap if colormap is None else colormap
    with sb.axes_style('whitegrid', rc={'grid.linestyle': 'dotted'}), sb.plotting_context('paper'):
        # print(sb.axes_style())
        # Initialize the figure
        strip_fig, axes = mp.pyplot.subplots(dpi=120, figsize=size or (10, len(df.index.unique())))
        set_scales(axes, xscale=xscale)
        if xbound is not None:
            axes.set_autoscalex_on(False)
            axes.set_xbound(*xbound)
            # axes.invert_xaxis()
        sb.despine(bottom=True, left=True)

        # Show each observation with a scatterplot
        sb.stripplot(data=df,
                     x=x, y=y, hue=hue,
                     hue_order=hue_order,
                     palette=colormap,
                     dodge=True, jitter=True,
                     alpha=.25, zorder=1)

        # Show the conditional means
        sb.pointplot(data=df,
                     x=x, y=y, hue=hue,
                     hue_order=hue_order,
                     palette=colormap,
                     dodge=.5, join=False,
                     markers='d', scale=.75, ci=None)

        # Improve the legend
        handles, labels = axes.get_legend_handles_labels()
        dist = int(len(labels)/2)
        handles, labels = handles[dist:], labels[dist:]
        if legend_labels is not None:
            if isinstance(legend_labels, list):
                labels = legend_labels
            else:
                labels = map(legend_labels, labels)

        axes.legend(handles, labels, title=legend_title or hue,
                    handletextpad=0, columnspacing=1,
                    loc=legend_loc, ncol=1, frameon=True)
        set_labels(axes, title=title, xlabel=xlabel, ylabel=ylabel, y_labels=y_labels)
        return strip_fig


def draw_score_stripplot(col, results, type_filter='all', metadata=None,
                         y_sort_by='name', hue_sort_by=None,
                         filename=None, **kwargs):
    sort_by = (y_sort_by if callable(y_sort_by)
               else None if not metadata or not isinstance(y_sort_by, str)
               else lambda row: row.task.apply(lambda t: getattr(metadata[t], y_sort_by)))
    plot_df = sort_dataframe(results.set_index(['type', 'task']), by=sort_by)
    df = (plot_df if type_filter == 'all'
          else plot_df[plot_df.index.get_loc(type_filter)])

    hue = 'framework'
    hues = sorted(df[hue].unique(), key=hue_sort_by)

    fig = draw_stripplot(
        df,
        x=col,
        y=df.index,
        hue=hue,
        # ylabel='Task',
        y_labels=task_labels(df.index.unique()),
        hue_order=hues,
        legend_title="Framework",
        **kwargs
    )
    if filename:
        savefig(fig, create_file("graphics", config.results_group, filename))
    return fig
