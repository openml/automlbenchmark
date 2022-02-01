import datetime as dt
import fnmatch
import itertools
import logging
import os
import shutil
import tempfile
import zipfile

from .core import Namespace
from .time import datetime_iso

log = logging.getLogger(__name__)

__no_export = set(dir())  # all variables defined above this are not exported


def to_mb(size_in_bytes):
    return size_in_bytes / (1 << 20)


def to_gb(size_in_bytes):
    return size_in_bytes / (1 << 30)


def normalize_path(path):
    return os.path.realpath(os.path.expanduser(path))


def split_path(path):
    dir, file = os.path.split(path)
    base, ext = os.path.splitext(file)
    return Namespace(dirname=dir, filename=file, basename=base, extension=ext)


def path_from_split(split, real_path=True):
    return os.path.join(os.path.realpath(split.dirname) if real_path else split.dirname,
                        split.basename)+('' if split.extension in [None, '']
                                         else split.extension if split.extension[0] == '.'
                                         else f".{split.extension}")


def dir_of(caller_file, rel_to_project_root=False):
    abs_path = os.path.realpath(os.path.dirname(caller_file))
    if rel_to_project_root:
        project_root = os.path.normpath(os.path.join(os.path.realpath(os.path.dirname(__file__)), '..', '..'))
        return os.path.relpath(abs_path, start=project_root)
    else:
        return abs_path


def list_all_files(paths, filter_=None):
    """
    :param paths: the directories to look into.
    :param filter_: None, or a predicate function returning True iff the file should be listed.
    """
    filter_ = filter_ or (lambda _: True)
    all_files = []
    paths = paths if isinstance(paths, list) else [paths]
    for path in paths:
        # path = normalize_path(path)
        if os.path.isdir(path):
            for root_dir, sub_dirs, files in os.walk(path):
                for name in files:
                    full_path = os.path.join(root_dir, name)
                    if filter_(full_path):
                        all_files.append(full_path)
        elif os.path.isfile(path):
            if filter_(path):
                all_files.append(path)
        else:
            log.warning("Skipping file `%s` as it doesn't exist.", path)

    return all_files


def touch(path, as_dir=False):
    path = normalize_path(path)
    if not os.path.exists(path):
        dirname, basename = (path, '') if as_dir else os.path.split(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        if basename:
            open(path, 'a').close()
    os.utime(path, times=None)
    return path


def backup_file(file_path):
    src_path = normalize_path(file_path)
    if not os.path.isfile(src_path):
        return
    p = split_path(src_path)
    mod_time = dt.datetime.utcfromtimestamp(os.path.getmtime(src_path))
    dest_name = ''.join([p.basename, '.', datetime_iso(mod_time, date_sep='', time_sep=''), p.extension])
    dest_dir = os.path.join(p.dirname, 'backup')
    touch(dest_dir, as_dir=True)
    dest_path = os.path.join(dest_dir, dest_name)
    shutil.copyfile(src_path, dest_path)
    log.debug('File `%s` was backed up to `%s`.', src_path, dest_path)


def _create_file_filter(filter_, default_value=True):
    matches = ((lambda _: default_value) if filter_ is None
               else filter_ if callable(filter_)
               else (lambda p: fnmatch.fnmatch(p, filter_)) if isinstance(filter_, str)
               else (lambda p: any(fnmatch.fnmatch(p, pat) for pat in filter_)) if isinstance(filter_, (list, tuple))
               else None)
    if matches is None:
        raise ValueError("filter should be None, a predicate function, a wildcard pattern or a list of those.")
    return matches


def file_filter(include=None, exclude=None):
    includes = _create_file_filter(include, True)
    excludes = _create_file_filter(exclude, False)
    return lambda p: includes(p) and not excludes(p)


def walk_apply(dir_path, apply, topdown=True, max_depth=-1, filter_=None):
    dir_path = normalize_path(dir_path)
    for dir, subdirs, files in os.walk(dir_path, topdown=topdown):
        if max_depth >= 0:
            depth = 0 if dir == dir_path else len(str.split(os.path.relpath(dir, dir_path), os.sep))
            if depth > max_depth:
                continue
        for p in itertools.chain(files, subdirs):
            path = os.path.join(dir, p)
            if filter_ is None or filter_(path):
                apply(path, isdir=(p in subdirs))


def clean_dir(dir_path, filter_=None):
    def delete(path, isdir):
        rm = filter_ is None or filter_(path)
        if not rm:
            return
        if isdir:
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)

    walk_apply(dir_path, delete, max_depth=0)


def zip_path(path, dest_archive, compression=zipfile.ZIP_DEFLATED, arc_path_format='short', filter_=None):
    path = normalize_path(path)
    if not os.path.exists(path): return
    with zipfile.ZipFile(dest_archive, 'w', compression) as zf:
        if os.path.isfile(path):
            in_archive = os.path.basename(path)
            zf.write(path, in_archive)
        elif os.path.isdir(path):
            def add_to_archive(file, isdir):
                if isdir: return
                in_archive = (os.path.relpath(file, path) if arc_path_format == 'short'
                              else os.path.relpath(file, os.path.dirname(path)) if arc_path_format == 'long'
                              else os.path.basename(file) is arc_path_format == 'flat'
                              )
                zf.write(file, in_archive)
            walk_apply(path, add_to_archive,
                       filter_=lambda p: (filter_ is None or filter_(p)) and not os.path.samefile(dest_archive, p))


class TmpDir:

    def __init__(self):
        self.tmp_dir = None

    def __enter__(self):
        self.tmp_dir = tempfile.mkdtemp()
        return self.tmp_dir

    def __exit__(self, *args):
        shutil.rmtree(self.tmp_dir)
        # pass


__all__ = [s for s in dir() if not s.startswith('_') and s not in __no_export]
