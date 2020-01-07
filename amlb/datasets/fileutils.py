import logging
import shutil
import tarfile
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen, urlretrieve
import zipfile

from ..utils import touch

log = logging.getLogger(__name__)

VALID_URLS = ("http", "https")


def is_valid_url(url):
    return urlparse(url).scheme in VALID_URLS


def url_exists(url):
    if not is_valid_url(url):
        return False
    head_req = Request(url, method='HEAD')
    try:
        with urlopen(head_req) as test:
            return test.status == 200
    except URLError as e:
        log.error(f"Cannot access url %s: %s", url, e)
        return False


def download_file(url, dest_path):
    touch(dest_path)
    # urlretrieve(url, filename=dest_path)
    with urlopen(url) as resp, open(dest_path, 'wb') as dest:
        shutil.copyfileobj(resp, dest)


def is_archive(path):
    return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)


def unarchive_file(path, dest_folder=None):
    # dest = dest_folder if dest_folder else os.path.dirname(path)
    dest = dest_folder if dest_folder else os.path.splitext(path)
    touch(dest, as_dir=True)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(path=dest_folder)
    elif tarfile.is_tarfile(path):
        with tarfile.TarFile(path) as tf:
            tf.extractall(path=dest_folder)
    return dest

