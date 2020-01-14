import report.config as config

config.renamed_frameworks = dict(
    constantpredictor_enc='constantpredictor'
)
config.excluded_frameworks = ['oboe']

# config.impute_missing_with = 'constantpredictor'

all_results_files = {
    'old': [
        "results_valid_ref.csv", "results_valid.csv",
        "results_small-2c1h_ref.csv", "results_small-2c1h.csv",
        "results_medium-4c1h_ref.csv", "results_medium-4c1h.csv",
        "results_medium-4c4h_ref.csv", "results_medium-4c4h.csv",
    ],
    '1h': [
        "results_small-8c1h_ref.csv", "results_small-8c1h.csv",
        "results_medium-8c1h_ref.csv", "results_medium-8c1h.csv",            
    ],
    '4h': [
        "results_small-8c4h_ref.csv", "results_small-8c4h.csv",
        "results_medium-8c4h_ref.csv", "results_medium-8c4h.csv",    
        "results_large-8c4h_ref.csv", "results_large-8c4h.csv",       
    ],
    '8h': [
        "results_large-8c8h_ref.csv", "results_large-8c8h.csv",        
    ]
}

config.results_group = '4h'
config.results_files = all_results_files[config.results_group]
config.tasks_sort_by = 'nrows'
