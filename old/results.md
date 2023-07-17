---
layout: page
title: Results
sidebar_link: true
sidebar_sort_order: 1
---

### Complete Results
[Complete results][reports] are also available in [csv] format or as simple [visualizations] for now.
We hope to provide interactive visualization in the future.

### Binary Results
A sample of the results obtained by running each framework over 10 folds for various durations each: for binary tasks, the plotted metric is AUC.
Smaller and medium datasets were trained for 1h and 4h. Larger datasets have been trained for 4h and 8h.

![Binary Results Stripplot 1h][binary_1h]

![Binary Results Stripplot 4h][binary_4h]

![Binary Results Stripplot 8h][binary_8h]

### Multiclass Results
A sample of the results obtained by running each framework over 10 folds for various durations each: for multiclass tasks, the plotted metric is logloss.
Smaller and medium datasets were trained for 1h and 4h. Larger datasets have been trained for 4h and 8h.

![Multiclass Results Stripplot 1h][multiclass_1h]

![Multiclass Results Stripplot 4h][multiclass_4h]

![Multiclass Results Stripplot 8h][multiclass_8h]

[binary_1h]:https://raw.github.com/openml/automlbenchmark/master/reports/graphics/1h/binary_results_stripplot.png
[multiclass_1h]:https://raw.github.com/openml/automlbenchmark/master/reports/graphics/1h/multiclass_results_stripplot.png
[binary_4h]:https://raw.github.com/openml/automlbenchmark/master/reports/graphics/4h/binary_results_stripplot.png
[multiclass_4h]:https://raw.github.com/openml/automlbenchmark/master/reports/graphics/4h/multiclass_results_stripplot.png
[binary_8h]:https://raw.github.com/openml/automlbenchmark/master/reports/graphics/8h/binary_results_stripplot.png
[multiclass_8h]:https://raw.github.com/openml/automlbenchmark/master/reports/graphics/8h/multiclass_results_stripplot.png
[reports]:https://github.com/openml/automlbenchmark/tree/master/reports
[csv]:https://github.com/openml/automlbenchmark/tree/master/reports/tables
[visualizations]:https://github.com/openml/automlbenchmark/tree/master/reports/graphics