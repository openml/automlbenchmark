import datetime as dt
import fnmatch
import logging
import os
import shutil
import tempfile

from .core import Namespace
from .time import datetime_iso

log = logging.getLogger(__name__)


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
                        split.basename)+split.extension


def dir_of(caller_file, rel_to_project_root=False):
    abs_path = os.path.realpath(os.path.dirname(caller_file))
    if rel_to_project_root:
        project_root = os.path.normpath(os.path.join(os.path.realpath(os.path.dirname(__file__)), '..', '..'))
        return os.path.relpath(abs_path, start=project_root)
    else:
        return abs_path


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
        else:
            log.warning("Skipping file `%s` as it doesn't exist.", path)

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


def touch(path, as_dir=False):
    path = normalize_path(path)
    if not os.path.exists(path):
        dirname, basename = (path, '') if as_dir else os.path.split(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        if basename:
            open(path, 'a').close()
    os.utime(path, times=None)


def backup_file(file_path):
    src_path = os.path.realpath(file_path)
    if not os.path.isfile(src_path):
        return
    p = split_path(src_path)
    mod_time = dt.datetime.utcfromtimestamp(os.path.getmtime(src_path))
    dest_name = ''.join([p.basename, '_', datetime_iso(mod_time, date_sep='', time_sep=''), p.extension])
    dest_dir = os.path.join(p.dirname, 'backup')
    touch(dest_dir, as_dir=True)
    dest_path = os.path.join(dest_dir, dest_name)
    shutil.copyfile(src_path, dest_path)
    log.debug('File `%s` was backed up to `%s`.', src_path, dest_path)


class TmpDir:

    def __init__(self):
        self.tmp_dir = None

    def __enter__(self):
        self.tmp_dir = tempfile.mkdtemp()
        return self.tmp_dir

    def __exit__(self, *args):
        shutil.rmtree(self.tmp_dir)
        # pass


