import matplotlib
matplotlib.use('agg')  # no need for tk
from .barplot import draw_score_barplot
from .heatmap import draw_score_heatmap
from .linplot import draw_score_parallel_coord
from .pointplot import draw_score_pointplot
from .stripplot import draw_score_stripplot
