import logging
import os
import shutil
import tarfile
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen, urlretrieve
import zipfile

from ..utils import touch

log = logging.getLogger(__name__)

SUPPORTED_SCHEMES = ("http", "https", "s3")


def s3_path_to_bucket_prefix(s3_path):
    s3_path_cleaned = s3_path.split('://', 1)[1]
    bucket, prefix = s3_path_cleaned.split('/', 1)

    return bucket, prefix


def is_s3_url(path):
    if type(path) != str:
        return False
    if (path[:2] == 's3') and ('://' in path[:6]):
        return True
    return False


def is_valid_url(url):
    return urlparse(url).scheme in SUPPORTED_SCHEMES


def url_exists(url):
    if not is_valid_url(url):
        return False
    if not is_s3_url(url):
        head_req = Request(url, method='HEAD')
        try:
            with urlopen(head_req) as test:
                return test.status == 200
        except URLError as e:
            log.error(f"Cannot access url %s: %s", url, e)
            return False
    else:
        import boto3
        from botocore.errorfactory import ClientError
        s3 = boto3.client('s3')
        bucket, key = s3_path_to_bucket_prefix(url)
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            log.error(f"Cannot access url %s: %s", url, e)
            return False


def download_file(url, dest_path):
    touch(dest_path)
    # urlretrieve(url, filename=dest_path)
    if not is_s3_url(url):
        with urlopen(url) as resp, open(dest_path, 'wb') as dest:
            shutil.copyfileobj(resp, dest)
    else:
        import boto3
        from botocore.errorfactory import ClientError
        s3 = boto3.resource('s3')
        bucket, key = s3_path_to_bucket_prefix(url)
        try:
            s3.Bucket(bucket).download_file(key, dest_path)
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                log.error("The object does not exist.")
            else:
                raise

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
        with tarfile.open(path) as tf:
            tf.extractall(path=dest_folder)
    return dest
