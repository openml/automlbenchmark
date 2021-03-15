from IPython import display as ipyd
from IPython.display import FileLink, FileLinks
import fnmatch
import os

import pandas as pd
from tabulate import tabulate

import amlb_report.config as config


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return repr(self.__dict__)

    def extend(self, **kwargs):
        clone = Namespace(**self.__dict__)
        clone.__dict__.update(**kwargs)
        return clone

    
def list_all_files(paths, include=None, exclude=None):
    """
    :param paths: the directories to look into.
    :param include: a UNIX path pattern for the files to be included in the result list
    :param exclude: a UNIX path pattern for the files to be excluded from the result list
    """
    all_files = []
    paths = paths if isinstance(paths, list) else [paths]
    for path in paths:
        # path = normalize_path(path)
        if os.path.isdir(path):
            for root_dir, sub_dirs, files in os.walk(path):
                for name in files:
                    all_files.append(os.path.join(root_dir, name))
        elif os.path.isfile(path):
            all_files.append(path)

    if include is not None:
        include = include if isinstance(include, list) else [include]
        included = []
        for pattern in include:
            included.extend(fnmatch.filter(all_files, pattern))
        all_files = [file for file in all_files if file in included]

    if exclude is not None:
        exclude = exclude if isinstance(exclude, list) else [exclude]
        excluded = []
        for pattern in exclude:
            excluded.extend(fnmatch.filter(all_files, pattern))
        all_files = [file for file in all_files if file not in excluded]

    return all_files


def create_file(*path_tokens):
    path = os.path.realpath(os.path.join(*path_tokens))
    if not os.path.exists(path):
        dirname, basename = os.path.split(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        if basename:
            open(path, 'a').close()
    return path


def display(fr, pretty=True, tab_format=None, float_format=None):
    float_format = config.ff if float_format is None else float_format
    with pd.option_context(
            'display.max_rows', len(fr),
            'display.float_format', lambda f: float_format % f
    ):
        if isinstance(fr, pd.Series):
            fr = fr.to_frame()
        if pretty and isinstance(fr, pd.DataFrame):
            if fr.index.is_unique:
                fr.style.set_properties(**{'vertical-align':'top'})
            ipyd.display(ipyd.HTML(fr.to_html()))
        elif tab_format and isinstance(fr, pd.DataFrame):
            print(tabulate(fr, fr.columns, tablefmt=tab_format))  # e.g. tab_format='grid'
        else:
            print(fr)


def sort_dataframe(df, by=None, axis=0):
    if axis == 1:
        cols = [col for col in df.columns]
        cols.sort(key=by) if by else cols.sort()
        return df[cols]
#         return df.sort_index(by, axis=1)
    elif by:
        tmp_sort = 'tmp_sort'
        tmp_df = df.reset_index()
        tmp_df = tmp_df.assign(tmp_sort=by)
        tmp_df.set_index([*df.index.names, tmp_sort], inplace=True)
        tmp_df.sort_index(level=tmp_sort, inplace=True)
        tmp_df.set_index(tmp_df.index.droplevel(tmp_sort), inplace=True)
        return tmp_df
    return df.sort_index()

