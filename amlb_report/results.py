"""
Loading results, formatting and adding columns.
result is the raw result metric computed from predictions at the end the benchmark: higher is always better!
 - For classification problems, it is usually auc for binary problems and negative log loss for multiclass problems.
 - For regression problems, it is usually negative rmse.
norm_result is a normalization of result on a [0, 1] scale, with {{zero_one_refs[0]}} scoring as 0 and {{zero_one_refs[1]}} scoring as 1.
imp_result for imputed results. Given a task and a framework:
 - if all folds results are missing, then no imputation occurs, and the result is nan for each fold.
 - if only some folds results are missing, then the missing result is imputed by the {{imp_framework}} result for this fold.
"""

import numpy as np
import pandas as pd

import amlb_report.config as config
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


def norm_result(row, res_col='result', zero_one_refs=None, ref_results=None, aggregation=None):
    if zero_one_refs is None:
        return row[res_col]

    def get_val(ref, default):
        try:
            if isinstance(ref, str):
                return (ref_results.loc[(ref_results.framework == ref)
                                        & (ref_results.task == row.task)]
                                       [res_col]
                                   .agg(aggregation) if aggregation
                        else ref_results.loc[(ref_results.framework == ref)
                                             & (ref_results.task == row.task)
                                             & (ref_results.fold == row.fold)]
                                            [res_col]
                                        .item())
            else:
                return ref
        except Exception:
            raise
            # return default

    zero, one = (get_val(ref, i) for i, ref in enumerate(zero_one_refs))
    norm_res = (row[res_col] - zero) / (one - zero)
    return (- norm_res if row['metric'].startswith("neg_") and one < 0 <= zero
            else norm_res)


def sorted_ints(arr):
    return sorted(list(map(int, arr[~np.isnan(arr)])))


def remove_duplicates(df, handling='fail'):
    if not df.index.is_unique:
        print("Duplicate entries:")
        display(df[df.index.duplicated(keep=False)].sort_values(by=df.index.names),
                pretty=False)
    assert df.index.is_unique or handling != 'fail'
    duplicated = (df.index.duplicated(keep='first') if handling == 'keep_first'
                  else df.index.duplicated(keep='last') if handling == 'keep_last'
                  else df.index.duplicated(keep=False) if handling == 'keep_none'
                  else np.full((len(df), 1), False))
    return df[~duplicated]


def prepare_results(results,
                    renamings=None,
                    exclusions=None,
                    imputation=None,
                    normalization=None,
                    ref_results=None,
                    duplicates_handling='fail',  # other options are 'keep_first', 'keep_last', 'keep_none'
                    include_metadata=False
                    ):
    if results is None or len(results) == 0:
        return None
    if isinstance(results, list):
        results = load_results(results) if all(isinstance(r, str) for r in results) else pd.concat(results, ignore_index=True)
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

    metadata = load_dataset_metadata(results) if include_metadata else {}

    done = results.set_index(['task', 'fold', 'framework'])
    done = remove_duplicates(done, handling=duplicates_handling)

    missing = (pd.DataFrame([(task, fold, framework, 'missing')
                             for task in tasks
                             for fold in range(config.nfolds)
                             for framework in frameworks
                             if (task, fold, framework) not in done.index],
                            columns=[*done.index.names, 'info'])
               .set_index(done.index.names))
    missing = remove_duplicates(missing, handling=duplicates_handling)
    failed = (results.loc[pd.notna(results['info'])]
              .set_index(done.index.names))
    failed = remove_duplicates(failed, handling=duplicates_handling)

    # extending the data frame
    results = results.append(missing.reset_index())
    if 'type' not in results:
        results['type'] = [task_prop(row, metadata, 'type') for _, row in results.iterrows()]

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

    if normalization is not None:
        res_col = 'imp_result' if imputation is not None else 'result'
        zero_one = normalization[0:2]
        aggr = normalization[2] if len(normalization) > 2 else None
        results['norm_result'] = [norm_result(row, res_col,
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
