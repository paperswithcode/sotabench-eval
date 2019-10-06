import hashlib
import gzip
import errno
import tarfile
import zipfile
import os
from tqdm import tqdm
from pathlib import Path


SOTABENCH_CACHE = Path.home() / ".cache"


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_batch_hash(output):
    """Calculate the hash for the output of a batch

    Args:
        output: data to be hashed
    """

    m = hashlib.sha256()
    m.update(str(output).encode("utf-8"))
    return m.hexdigest()


def change_root_if_server(root, server_root):
    """
    :param root: string with a user-specified root
    :param server_root: string with a server root
    :return: server_root if SOTABENCH_SERVER env variable is set, else root
    """
    check_server = os.environ.get("SOTABENCH_SERVER")

    if check_server == 'true':
        return server_root

    return root


def is_server():
    """
    If true, uses env variable SOTABENCH_SERVER to determine whether code is being run on the server

    You can use this function for your control flow for server specific settings - e.g. the data paths.
    :return:
    """
    if os.environ.get("SOTABENCH_SERVER") == 'true':
        return True
    else:
        return False


def set_env_on_server(env_name, value):
    """
    If run on sotabench server, sets an environment variable with a given name to value (casted to str).

    :param env_name: environment variable name
    :param value: value to set if executed on sotabench
    :return: whether code is being run on the server
    """
    if is_server():
        os.environ[env_name] = str(value)
        return True
    return False


def get_max_memory_allocated(device='cuda'):
    try:
        import torch
        max_mem = torch.cuda.max_memory_allocated(device='cuda')
        torch.cuda.reset_max_memory_allocated(device='cuda')
        return max_mem
    except ImportError:
        return None

# below utilities are taken from the torchvision repository


def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root - utility function taken from torchvision repository
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    # downloads file
    if check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater()
            )
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater()
                )
            else:
                raise e


def _is_tar(filename):
    return filename.endswith(".tar")


def _is_targz(filename):
    return filename.endswith(".tar.gz")


def _is_gzip(filename):
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename):
    return filename.endswith(".zip")


def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)
