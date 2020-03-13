

def savefig(fig, path):
    fig.savefig(path, bbox_inches='tight')


def task_labels(index):
    max_length = 16
    return (index.droplevel('type')
            .map(lambda x: x if len(x) <= max_length else u'{}…'.format(x[:max_length-1]))
            .values)


def set_labels(axes,
               title=None,
               xlabel=None, ylabel=None,
               x_labels=None, y_labels=None,
               x_tick_params=None, y_tick_params=None,
               legend_title=None):

    axes.set_title('' if not title else title, fontsize='xx-large')
    axes.set_xlabel('' if not xlabel else xlabel, fontsize='x-large')
    axes.set_ylabel('' if not ylabel else ylabel, fontsize='x-large')
    if not x_tick_params:
        x_tick_params = {}
    if not y_tick_params:
        y_tick_params = {}
    axes.tick_params(axis='x', labelsize='x-large', **x_tick_params)
    axes.tick_params(axis='y', labelsize='x-large', **y_tick_params)
    if x_labels is not None:
        axes.set_xticklabels(x_labels)
    if y_labels is not None:
        axes.set_yticklabels(y_labels)
    legend = axes.get_legend()
    if legend:
        legend_title = legend_title or legend.get_title().get_text()
        legend.set_title(legend_title, prop=dict(size='x-large'))
        for text in legend.get_texts():
            text.set_fontsize('x-large')


def set_scales(axes, xscale=None, yscale=None):
    if isinstance(xscale, str):
        axes.set_xscale(xscale)
    elif isinstance(xscale, tuple):
        axes.set_xscale(xscale[0], **xscale[1])
    if isinstance(yscale, str):
        axes.set_yscale(yscale)
    elif isinstance(yscale, tuple):
        axes.set_yscale(yscale[0], **yscale[1])
