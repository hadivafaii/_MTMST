import re
import os
import json
import h5py
import torch
import pickle
import joblib
import shutil
import random
import pathlib
import inspect
import logging
import argparse
import warnings
import operator
import functools
import itertools
import collections
import numpy as np
import pandas as pd
import bottleneck as bn
from rich import print
from scipy import fft as sp_fft
from scipy import signal as sp_sig
from scipy import linalg as sp_lin
from scipy import stats as sp_stats
from scipy import ndimage as sp_img
from scipy import optimize as sp_optim
from scipy.spatial import distance as sp_dist
from scipy.spatial.transform import Rotation
from sklearn.preprocessing import Normalizer
from numpy.ma import masked_where as mwh
from prettytable import PrettyTable
from os.path import join as pjoin
from datetime import datetime
from tqdm import tqdm
from typing import *


def true(s: str):
	return str(s).lower() == 'true'


def escape_parenthesis(s: str):
	print(s.replace('(', '\(').replace(')', '\)'))


def to_np(x: Union[torch.Tensor, np.ndarray]):
	if isinstance(x, np.ndarray):
		return x
	return x.data.cpu().numpy()


def flatten_arr(
		x: np.ndarray,
		ndim_end: int = 1,
		ndim_start: int = 0, ):
	shape = x.shape
	assert 0 <= ndim_end <= len(shape)
	assert 0 <= ndim_start <= len(shape)
	if ndim_end + ndim_start >= len(shape):
		return x

	shape_flat = shape[:ndim_start] + (-1,)
	for i, d in enumerate(shape):
		if i >= len(shape) - ndim_end:
			shape_flat += (d,)
	return x.reshape(shape_flat)


def avg(
		x: np.ndarray,
		ndim_end: int = 2,
		ndim_start: int = 0,
		fn: Callable = bn.nanmean, ) -> np.ndarray:
	dims = range(ndim_start, x.ndim - ndim_end)
	dims = sorted(dims, reverse=True)
	for axis in dims:
		x = fn(x, axis=axis)
	return x


def cat_map(x: list, axis: int = 0):
	out = []
	for a in x:
		if len(a):
			out.append(np.concatenate(
				a, axis=axis))
		else:
			out.append(a)
	return out


def get_tval(
		dof: int = 9,
		ci: float = 0.95,
		two_sided: bool = True, ):
	if two_sided:
		ci = (1 + ci) / 2
	return sp_stats.t.ppf(ci, dof)


def contig_segments(mask: np.ndarray):
	censored = np.where(mask == 0)[0]
	looper = itertools.groupby(
		enumerate(censored),
		lambda t: t[0] - t[1],
	)
	segments = []
	for k, g in looper:
		s = map(operator.itemgetter(1), g)
		segments.append(list(s))
	return segments


def unique_idxs(
		obj: np.ndarray,
		filter_zero: bool = True, ):
	idxs = pd.DataFrame(obj.flat)
	idxs = idxs.groupby([0]).indices
	if filter_zero:
		idxs.pop(0, None)
	return idxs


def all_equal(iterable):
	g = itertools.groupby(iterable)
	return next(g, True) and not next(g, False)


def np_nans(shape: Union[int, Iterable[int]]):
	if isinstance(shape, np.ndarray):
		shape = shape.shape
	arr = np.empty(shape, dtype=float)
	arr[:] = np.nan
	return arr


def make_logger(
		name: str,
		path: str,
		level: int,
		module: str = None, ) -> logging.Logger:
	os.makedirs(path, exist_ok=True)
	logger = logging.getLogger(module)
	logger.setLevel(level)
	file = pjoin(path, f"{name}.log")
	file_handler = logging.FileHandler(file)
	formatter = logging.Formatter(
		'%(asctime)s : %(levelname)s : %(name)s : %(message)s')
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	return logger


def get_rng(
		x: Union[int, np.random.Generator, random.Random] = 42,
		use_np: bool = True, ):
	if isinstance(x, int):
		if use_np:
			return np.random.default_rng(seed=x)
		else:
			return random.Random(x)
	elif isinstance(x, (np.random.Generator, random.Random)):
		return x
	else:
		print('Warning, invalid random state. returning default')
		return np.random.default_rng(seed=42)


def setup_kwargs(defaults, kwargs):
	if not kwargs:
		return defaults
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v
	return kwargs


def get_default_params(f: Callable):
	params = inspect.signature(f).parameters
	return {
		k: p.default for
		k, p in params.items()
	}


def filter_kwargs(fn, kw: dict = None):
	if not kw:
		return {}
	try:
		params = inspect.signature(fn).parameters
		return {
			k: v for k, v
			in kw.items()
			if k in params
		}
	except ValueError:
		return kw


def save_obj(
		obj: Any,
		file_name: str,
		save_dir: str,
		mode: str = None,
		verbose: bool = True, ):
	_allowed_modes = [
		'npy', 'df',
		'pkl', 'joblib',
		'html', 'json', 'txt',
	]
	_ext = file_name.split('.')[-1]
	if _ext in _allowed_modes:
		mode = _ext
	else:
		if mode is None:
			msg = 'invalid file extension: '
			msg += f"{_ext}, mode: {mode}"
			raise RuntimeError(msg)
		else:
			file_name = f"{file_name}.{mode}"
	assert mode in _allowed_modes, \
		f"available modes:\n{_allowed_modes}"

	path = pjoin(save_dir, file_name)
	op_mode = 'w' if mode in ['html', 'json', 'txt'] else 'wb'
	with open(path, op_mode) as f:
		if mode == 'npy':
			np.save(f.name, obj)
		elif mode == 'df':
			pd.to_pickle(obj, f.name)
		elif mode == 'pkl':
			# noinspection PyTypeChecker
			pickle.dump(obj, f)
		elif mode == 'joblib':
			joblib.dump(obj, f)
		elif mode == 'html':
			f.write(obj)
		elif mode == 'json':
			json.dump(obj, f, indent=4)
		elif mode == 'txt':
			for line in obj:
				f.write(line)
		else:
			raise RuntimeError(mode)
	if verbose:
		print(f"[PROGRESS] '{file_name}' saved at\n{save_dir}")
	return path


def merge_dicts(
		dict_list: List[dict],
		verbose: bool = False, ) -> Dict[str, list]:
	merged = collections.defaultdict(list)
	dict_items = map(operator.methodcaller('items'), dict_list)
	iterable = itertools.chain.from_iterable(dict_items)
	kws = {
		'leave': False,
		'disable': not verbose,
		'desc': "...merging dicts",
	}
	for k, v in tqdm(iterable, **kws):
		merged[k].extend(v)
	return dict(merged)


def base2(number: int):
	b = np.base_repr(number, base=2)
	if len(b) == 1:
		return 0, 0, int(b)
	elif len(b) == 2:
		j, k = b
		return 0, int(j), int(k)
	elif len(b) == 3:
		i, j, k = b
		return int(i), int(j), int(k)
	else:
		return b


def now(include_hour_min: bool = False):
	s = "%Y_%m_%d"
	if include_hour_min:
		s += ",%H:%M"
	return datetime.now().strftime(s)
