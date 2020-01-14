
nfolds = 10
ff = '%.6g'
colormap = 'tab10'
# colormap = 'Set2'
# colormap = 'Dark2'

renamed_frameworks = dict()
excluded_frameworks = []

binary_score_label = 'AUC'
multiclass_score_label = 'logloss'

# impute_missing_with = 'constantpredictor'
impute_missing_with = 'randomforest'
zero_one_refs = ('constantpredictor', 'tunedrandomforest')

tasks_sort_by = 'nrows'

results_files = []
results_group = ''
