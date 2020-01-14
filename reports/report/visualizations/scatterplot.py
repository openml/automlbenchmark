import matplotlib as mp
import seaborn as sb

import report.config as config
from ..util import create_file, sort_dataframe
from .util import savefig, set_scales, set_labels, task_labels


def draw_stripplot(df, x, y, hue,
                   xscale='linear', xbound=None,
                   xlabel=None, ylabel=None, y_labels=None, title=None,
                   legend_title=None, legend_loc='best', colormap=None):
    colormap = config.colormap if colormap is None else colormap
    with sb.axes_style('whitegrid', rc={'grid.linestyle': 'dotted'}), sb.plotting_context('paper'):
        # print(sb.axes_style())
        # Initialize the figure
        strip_fig, axes = mp.pyplot.subplots(dpi=120, figsize=(10, len(df.index.unique())))
        set_scales(axes, xscale=xscale)
        if xbound is not None:
            axes.set_autoscalex_on(False)
            axes.set_xbound(*xbound)
            # axes.invert_xaxis()
        sb.despine(bottom=True, left=True)

        # Show each observation with a scatterplot
        sb.stripplot(x=x, y=y, hue=hue,
                     data=df, dodge=True, jitter=True, palette=colormap,
                     alpha=.25, zorder=1)

        # Show the conditional means
        sb.pointplot(x=x, y=y, hue=hue,
                     data=df, dodge=.5, join=False, palette=colormap,
                     markers='d', scale=.75, ci=None)

        # Improve the legend
        handles, labels = axes.get_legend_handles_labels()
        dist = int(len(labels)/2)
        axes.legend(handles[dist:], labels[dist:], title=legend_title or hue,
                    handletextpad=0, columnspacing=1,
                    loc=legend_loc, ncol=1, frameon=True)
        set_labels(axes, title=title, xlabel=xlabel, ylabel=ylabel, y_labels=y_labels)
        return strip_fig


def draw_score_stripplot(col, results, type_filter='all', metadata=None,
                         y_sort_by='name', filename=None, **kwargs):
    sort_by = (None if not metadata
               else lambda row: row.task.apply(lambda t: getattr(metadata[t], y_sort_by)))
    scatterplot_df = sort_dataframe(results.set_index(['type', 'task']), by=sort_by)
    df = scatterplot_df if type_filter == 'all' \
        else scatterplot_df[scatterplot_df.index.get_loc(type_filter)]
    fig = draw_stripplot(
        df,
        x=col,
        y=df.index,
        hue='framework',
        # ylabel='Task',
        y_labels=task_labels(df.index.unique()),
        legend_title="Framework",
        **kwargs
    )
    if filename is not None:
        savefig(fig, create_file("graphics", config.results_group, filename))
    return fig
