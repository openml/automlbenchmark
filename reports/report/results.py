"""
Loading results, formatting and adding columns
result is the raw result metric computed from predictions at the end the benchmark. For classification problems, it is usually auc for binomial classification and logloss for multinomial classification.
score ensures a standard comparison between tasks: higher is always better.
norm_score is a normalization of score on a [0, 1] scale, with {{zero_one_refs[0]}} score as 0 and {{zero_one_refs[1]}} score as 1.
imp_result and imp_score for imputed results/scores. Given a task and a framework:
if all folds results/scores are missing, then no imputation occurs, and the result is nan for each fold.
if only some folds results/scores are missing, then the missing result is imputed by the {{impute_missing_with}} result for this fold.
"""

import numpy as np
import pandas as pd

import report.config as config
from .metadata import load_dataset_metadata
from .util import Namespace, display


def load_results(files):
    return pd.concat([pd.read_csv(file) for file in files], ignore_index=True)


def task_prop(row, metadata, prop):
    return getattr(metadata.get(row.task), prop)


def impute_result(row, results, res_col='result',
                  imp_framework=None, imp_results=None,
                  imp_value=None, aggregation=None):
    if pd.notna(row[res_col]):
        return row[res_col]
    # if all folds are failed or missing, don't impute
    if pd.isna(results.loc[(results.task == row.task)
                           & (results.framework == row.framework)][res_col]).all():
        return np.nan
    if imp_framework is not None:
        # impute with ref framework corresponding value
        imp_results = results if imp_results is None else imp_results
        return (imp_results.loc[(imp_results.framework == imp_framework)
                                & (imp_results.task == row.task)]
                               [res_col]
                           .agg(aggregation) if aggregation
                else imp_results.loc[(imp_results.framework == imp_framework)
                                     & (imp_results.task == row.task)
                                     & (imp_results.fold == row.fold)]
                                    [res_col]
                                .item())
    return imp_value


def imputed(row):
    return pd.isna(row.result) and pd.notna(row.imp_result)


fit_metrics = ['auc', 'acc', 'r2']


def metric_type(row, res_col='result'):
    return 'fit' if any([row[res_col] == getattr(row, m, None) for m in fit_metrics]) else 'loss'


def score(row, res_col='result'):
    return (row[res_col] if row['metric_type'] == 'fit'
            else - row[res_col])


def norm_score(row, score_col='score',
               zero_one_refs=None, ref_results=None,
               aggregation=None):
    if zero_one_refs is None:
        return row[score_col]

    def get_val(ref, default):
        try:
            if isinstance(ref, str):
                return (ref_results.loc[(ref_results.framework == ref)
                                        & (ref_results.task == row.task)]
                                       [score_col]
                                   .agg(aggregation) if aggregation
                        else ref_results.loc[(ref_results.framework == ref)
                                             & (ref_results.task == row.task)
                                             & (ref_results.fold == row.fold)]
                                            [score_col]
                                        .item())
            else:
                return ref
        except Exception:
            raise
            # return default

    zero, one = (get_val(ref, i) for i, ref in enumerate(zero_one_refs))
    rel_score = (row[score_col] - zero) / (one - zero)
    return (- rel_score if row['metric_type'] == 'loss' and one < 0 <= zero
            else rel_score)


def sorted_ints(arr):
    return sorted(list(map(int, arr[~np.isnan(arr)])))


def prepare_results(results_files,
                    renamings=None,
                    exclusions=None,
                    imputation=None,
                    normalization=None,
                    ref_results=None
                    ):
    results = load_results(results_files)
    if renamings:
        results.replace(renamings, inplace=True)
    if exclusions:
        results = results.loc[~results.framework.isin(exclusions)]
    results.task = results.task.str.lower()
    results.framework = results.framework.str.lower()
    results.fold = results.fold.apply(int)

    frameworks = results.framework.unique()
    frameworks.sort()

    tasks = results.task.unique()
    tasks.sort()

    folds = results.fold.unique()

    metadata = load_dataset_metadata(results)

    done = results.set_index(['task', 'fold', 'framework'])
    if not done.index.is_unique:
        print("Duplicate entries:")
        display(done[done.index.duplicated(keep=False)].sort_values(by=done.index.names),
                pretty=False)
    assert done.index.is_unique

    missing = (pd.DataFrame([(task, fold, framework, 'missing')
                             for task in tasks
                             for fold in range(config.nfolds)
                             for framework in frameworks
                             if (task, fold, framework) not in done.index],
                            columns=[*done.index.names, 'info'])
               .set_index(done.index.names))
    assert missing.index.is_unique
    failed = (results.loc[pd.notna(results['info'])]
              .set_index(done.index.names))
    assert failed.index.is_unique

    # extending the data frame
    results = results.append(missing.reset_index())
    results['type'] = [task_prop(row, metadata, 'type') for _, row in results.iterrows()]
    results['metric_type'] = [metric_type(row) for _, row in results.iterrows()]
    results['score'] = [score(row) for _, row in results.iterrows()]

    if ref_results is None:
        ref_results = results

    if imputation is not None:
        imp_fr = imp_val = aggr = None
        if isinstance(imputation, tuple):
            imp_fr, aggr = imputation
        elif isinstance(imputation, str):
            imp_fr = imputation
        else:
            imp_val = imputation
        results['imp_result'] = [impute_result(row, results,
                                               imp_framework=imp_fr, imp_results=ref_results,
                                               imp_value=imp_val, aggregation=aggr)
                                 for _, row in results.iterrows()]
        results['imp_score'] = [impute_result(row, results, 'score',
                                              imp_framework=imp_fr, imp_results=ref_results,
                                              imp_value=imp_val, aggregation=aggr)
                                for _, row in results.iterrows()]

    if normalization is not None:
        score_col = 'imp_score' if imputation is not None else 'score'
        zero_one = normalization[0:2]
        aggr = normalization[2] if len(normalization) > 2 else None
        results['norm_score'] = [norm_score(row, score_col,
                                            zero_one_refs=zero_one, ref_results=ref_results, aggregation=aggr)
                                 for _, row in results.iterrows()]

    return Namespace(
        results=results,
        frameworks=frameworks,
        tasks=tasks,
        folds=folds,
        metadata=metadata,
        done=done,
        missing=missing,
        failed=failed
    )
