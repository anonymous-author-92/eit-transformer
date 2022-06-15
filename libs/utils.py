import gc
import os
import sys
import argparse
import math
import pickle
import copy
import random as rd
from collections import defaultdict
from contextlib import contextmanager
from datetime import date
from time import time

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams, tri
import numpy as np
import pandas as pd
import psutil
import platform
import subprocess
import re
import torch
import torch.nn as nn

import h5py
from scipy.io import loadmat

try:
    import plotly.express as px
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    import plotly.io as pio
except ImportError as e:
    print('Plotly not installed. Plotly is used to plot meshes')

import ctypes
import ctypes.util

libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
libc.mount.argtypes = (ctypes.c_char_p, ctypes.c_char_p,
                       ctypes.c_char_p, ctypes.c_ulong, ctypes.c_char_p)


def get_size(bytes, suffix='B'):
    ''' 
    by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MiB'
        1253656678 => '1.17GiB'
    '''
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(bytes) < 1024.0:
            return f"{bytes:3.2f} {unit}{suffix}"
        bytes /= 1024.0
    return f"{bytes:3.2f} 'Yi'{suffix}"


def get_file_size(filename):
    file_size = os.stat(filename)
    return get_size(file_size.st_size)


def get_processor_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command = "sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).strip()
        for line in all_info.decode("utf-8").split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1)


def get_system():
    print("="*40, "CPU Info", "="*40)
    # number of cores
    print("Device name       :", get_processor_name())
    print("Physical cores    :", psutil.cpu_count(logical=False))
    print("Total cores       :", psutil.cpu_count(logical=True))
    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    print(f"Max Frequency    : {cpufreq.max:.2f} Mhz")
    print(f"Min Frequency    : {cpufreq.min:.2f} Mhz")
    print(f"Current Frequency: {cpufreq.current:.2f} Mhz")

    print("="*40, "Memory Info", "="*40)
    # get the memory details
    svmem = psutil.virtual_memory()
    print(f"Total     : {get_size(svmem.total)}")
    print(f"Available : {get_size(svmem.available)}")
    print(f"Used      : {get_size(svmem.used)}")

    print("="*40, "Software Info", "="*40)
    print('Python     : ' + sys.version.split('\n')[0])
    print('Numpy      : ' + np.__version__)
    print('Pandas     : ' + pd.__version__)
    print('PyTorch    : ' + torch.__version__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        print("="*40, "GPU Info", "="*40)
        print(f'Device     : {torch.cuda.get_device_name(0)}')
        print(f"{'Mem total': <15}: {round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB")
        print(
            f"{'Mem allocated': <15}: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB")
        print(
            f"{'Mem cached': <15}: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB")

    print("="*30, "system info print done", "="*30)


def get_seed(s, printout=True, cudnn=True):
    # rd.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)
    np.random.seed(s)
    # pd.core.common.random_state(s)
    # Torch
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    if cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

    message = f'''
    os.environ['PYTHONHASHSEED'] = str({s})
    numpy.random.seed({s})
    torch.manual_seed({s})
    torch.cuda.manual_seed({s})
    '''
    if cudnn:
        message += f'''
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        '''

    if torch.cuda.is_available():
        message += f'''
        torch.cuda.manual_seed_all({s})
        '''
    if printout:
        print("\n")
        print(f"The following code snippets have been run.")
        print("="*50)
        print(message)
        print("="*50)


@contextmanager
def simple_timer(title):
    t0 = time()
    yield
    print("{} - done in {:.1f} seconds.\n".format(title, time() - t0))


class Colors:
    """Defining Color Codes to color the text displayed on terminal.
    """

    red = "\033[91m"
    green = "\033[92m"
    yellow = "\033[93m"
    blue = "\033[94m"
    magenta = "\033[95m"
    end = "\033[0m"


def color(string: str, color: Colors = Colors.yellow) -> str:
    return f"{color}{string}{Colors.end}"


@contextmanager
def timer(label: str, compact=False) -> None:
    '''
    https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/203020#1111022
    print 
    1. the time the code block takes to run
    2. the memory usage.
    '''
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2. ** 30
    start = time()  # Setup - __enter__
    if not compact:
        print(color(f"{label}: start at {start:.2f};", color=Colors.blue))
        print(
            color(f"LOCAL RAM USAGE AT START: {m0:.2f} GB", color=Colors.green))
        try:
            yield  # yield to body of `with` statement
        finally:  # Teardown - __exit__
            m1 = p.memory_info()[0] / 2. ** 30
            delta = m1 - m0
            sign = '+' if delta >= 0 else '-'
            delta = math.fabs(delta)
            end = time()
            print(color(
                f"{label}: done at {end:.2f} ({end - start:.6f} secs elapsed);", color=Colors.blue))
            print(color(
                f"LOCAL RAM USAGE AT END: {m1:.2f}GB ({sign}{delta:.2f}GB)", color=Colors.green))
            print('\n')
    else:
        yield
        print(
            color(f"{label} - done in {time() - start:.6f} seconds. \n", color=Colors.blue))


def get_memory(num_var=10):
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in globals().items()), key=lambda x: -x[1])[:num_var]:
        print(color(f"{name:>30}:", color=Colors.green),
              color(f"{get_size(size):>8}", color=Colors.magenta))


def find_files(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        for _file in files:
            if name in _file:
                result.append(os.path.join(root, _file))
    return sorted(result)


def print_file_size(files):
    for file in files:
        size = get_file_size(file)
        filename = file.split('/')[-1]
        filesize = get_file_size(file)
        print(color(f"{filename:>30}:", color=Colors.green),
              color(f"{filesize:>8}", color=Colors.magenta))


@contextmanager
def trace(title: str):
    t0 = time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2. ** 30
    yield
    m1 = p.memory_info()[0] / 2. ** 30
    delta = m1 - m0
    sign = '+' if delta >= 0 else '-'
    delta = math.fabs(delta)
    print(f"[{m1:.1f}GB ({sign}{delta:.3f}GB): {time() - t0:.2f}sec] {title} ", file=sys.stderr)


def get_cmap(n, cmap='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(cmap, n)


def get_date():
    today = date.today()
    return today.strftime("%b-%d-%Y")

def argmax(lst):
    '''
    Taken from https://stackoverflow.com/a/31105620/622119
    License: CC BY-SA 3.0.
    '''
    return lst.index(max(lst))


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = 0
    for p in model_parameters:
        num_params += np.prod(p.size()+(2,) if p.is_complex() else p.size())
    return num_params


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            f'Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df


def save_pickle(var, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(var, f)


def load_pickle(load_path):
    with open(load_path, 'rb') as f:
        u = pickle.load(f)
    return u


def default(value, d):
    '''
    helper taken from https://github.com/lucidrains/linear-attention-transformer
    '''
    return d if value is None else value


class DotDict(dict):
    """
    https://stackoverflow.com/a/23689767/622119
    https://stackoverflow.com/a/36968114/622119
    dot.notation access to dictionary attributes
    """

    def __getattr__(self, attr):
        return self.get(attr)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self


def mount(source, target, fs, options=''):
    ret = libc.mount(source.encode(), target.encode(),
                     fs.encode(), 0, options.encode())
    if ret < 0:
        errno = ctypes.get_errno()
        raise OSError(
            errno, f"Error mounting {source} ({fs}) on {target} with options '{options}': {os.strerror(errno)}")

def clones(module, N):
    '''
    Input:
        - module: nn.Module obj
    Output:
        - zip identical N layers (not stacking)

    Refs:
        - https://nlp.seas.harvard.edu/2018/04/03/attention.html
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def csr_to_sparse(M):
    '''
    Input:
        M: csr_matrix
    Output:
        torch sparse tensor

    Another implementation can be found in
    https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py
    def sparse_mx_to_torch_sparse_tensor(sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    '''
    n, m = M.shape
    coo_ = M.tocoo()
    ix = torch.LongTensor([coo_.row, coo_.col])
    M_t = torch.sparse.FloatTensor(ix,
                                   torch.from_numpy(M.data).float(),
                                   [n, m])
    return M_t


if __name__ == "__main__":
    get_system()
    get_memory()
