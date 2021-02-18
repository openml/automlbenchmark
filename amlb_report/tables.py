"""
Averaging using arithmetic mean over fold result or score. In following summaries, if not mentioned otherwise, the means are computed over imputed results/scores. Given a task and a framework:

if all folds results/scores are missing, then no imputation occured, and the mean result is nan.
if only some folds results/scores are missing, then the amount of imputed results that contributed to the mean are displayed between parenthesis.
"""

import functools as ft

import numpy as np
import pandas as pd

import amlb_report.config as config
from .results import imputed
from .util import create_file, display


def add_imputed_mark(values, imp, val_type=float, val_format=None):
    formats = dict(float="{:,.6g}{}", int="{0:d}{}", str="{}{}")
    format_value = (val_format if val_format is not None
                    else lambda *val: formats[val_type.__name__].format(*val))
    return (values.astype(object)
            .combine(imp,
                     lambda val, imp: format_value(val, " ({:.0g})".format(imp) if imp else '')))


def render_summary(col, results, show_imputations=True, filename=None, float_format=None):
    float_format = config.ff if float_format is None else float_format
    res_group = results.groupby(['type', 'task', 'framework'])
    df = res_group[col].mean().unstack()
    if show_imputations and 'imp_result' in results.columns:
        imputed_df = (res_group['result', 'imp_result']
                      .apply(lambda df: sum(imputed(row) for _, row in df.iterrows()))
                      .unstack())
        df = df.combine(imputed_df, ft.partial(add_imputed_mark,
                                               val_format=lambda *v: (float_format+"%s") % tuple(v)))
    display(df, float_format=float_format)
    if filename:
        df.to_csv(create_file("tables", config.results_group, filename), float_format=float_format)
    return df


def rank(scores):
    sorted_scores = pd.Series(scores.unique()).sort_values(ascending=False)
    ranks = pd.Series(index=scores.index)
    for idx, value in scores.items():
        try:
            ranks.at[idx] = np.where(sorted_scores == value)[0][0]+1
        except IndexError:
            ranks.at[idx] = np.nan
    return ranks


def render_leaderboard(col, results, aggregate=False, show_imputations=True, filename=None):
    res_group = results.groupby(['type', 'task', 'framework'])
    df = (res_group[col].mean().unstack() if aggregate
          else results.pivot_table(index=['type','task', 'fold'], columns='framework', values=col))
    df = (df.apply(rank, axis=1, result_type='broadcast')
          .astype(object))
    if show_imputations and 'imp_result' in results.columns:
        imputed_df = (res_group['result', 'imp_result']
                      .apply(lambda df: sum(imputed(row) for _, row in df.iterrows()))
                      .unstack())
        df = df.combine(imputed_df, add_imputed_mark)
    display(df)
    if filename:
        df.to_csv(create_file("tables", config.results_group, filename), float_format='%.f')
    return df

