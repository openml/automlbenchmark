import report.config as config

config.renamed_frameworks = dict(
    constantpredictor_enc='constantpredictor'
)
config.excluded_frameworks = ['oboe']

# config.impute_missing_with = 'constantpredictor'

config.results_group = ''
config.results_files = []
config.tasks_sort_by = 'nrows'
config.colormap = 'colorblind'
