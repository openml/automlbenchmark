import logging
import os
import shutil
import tarfile
import boto3
from botocore.errorfactory import ClientError
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile

from ..utils import touch

log = logging.getLogger(__name__)


class FileHandler:
    def exists(self, url): pass
    def download(self, url, dest_path): pass


class HttpHandler(FileHandler):
    def exists(self, url):
        head_req = Request(url, method='HEAD')
        try:
            with urlopen(head_req) as test:
                return test.status == 200
        except URLError as e:
            log.error(f"Cannot access url %s: %s", url, e)
            return False
    
    def download(self, url, dest_path):
        touch(dest_path)
        with urlopen(url) as resp, open(dest_path, 'wb') as dest:
            shutil.copyfileobj(resp, dest)


class S3Handler(FileHandler):
    def exists(self, url):
        s3 = boto3.client('s3')
        bucket, key = self._s3_path_to_bucket_prefix(url)
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            log.error(f"Cannot access url %s: %s", url, e)
            return False
        
    def download(self, url, dest_path):
        touch(dest_path)
        s3 = boto3.resource('s3')
        bucket, key = self._s3_path_to_bucket_prefix(url)
        try:
            s3.Bucket(bucket).download_file(key, dest_path)
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                log.error("The object does not exist.")
            else:
                raise
        
    def _s3_path_to_bucket_prefix(self, s3_path):
        s3_path_cleaned = s3_path.split('://', 1)[1]
        bucket, prefix = s3_path_cleaned.split('/', 1)
        return bucket, prefix


scheme_handlers = dict(
    http=HttpHandler(),
    https=HttpHandler(),
    s3=S3Handler(),
    s3a=S3Handler(),
    s3n=S3Handler(),
)

SUPPORTED_SCHEMES = list(scheme_handlers.keys())


def get_file_handler(url):
    return scheme_handlers[urlparse(url).scheme]


def is_valid_url(url):
    return urlparse(url).scheme in SUPPORTED_SCHEMES


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
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tf, path=dest_folder)
    return dest
