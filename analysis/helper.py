from utils.plotting import *
from sklearn import metrics as sk_metric
from sklearn import neighbors as sk_neigh
from sklearn import inspection as sk_inspect
from sklearn import linear_model as sk_linear
from sklearn import decomposition as sk_decomp
from sklearn import model_selection as sk_modselect


def compute_ed(s):
	return sum(s)**2 / sum(s**2)


def normalize_global(x: np.ndarray, mu: float, sd: float):
	if x is None:
		return x
	return (x - mu) / sd


def max_r2(y: np.ndarray):
	"""
	:param y: neural responses, shape = (nc, ntrials, nt)
	:return: maximum attainable r2 score
	"""
	n_trials = y.shape[1]
	response_power = np.nanvar(np.nanmean(y, 1), -1)
	signal_power = (
		n_trials * response_power -
		np.nanmean(np.nanvar(y, -1), 1)
	) / (n_trials - 1)
	return signal_power / response_power


def discrete_mutual_info(
		z: np.ndarray,
		g: np.ndarray,
		axis: int = 1,
		n_bins: int = 20,
		parallel: bool = True,
		n_jobs: int = -1, ):
	assert axis in [0, 1]
	assert g.ndim == z.ndim == 2
	shape = (g.shape[axis], z.shape[axis])
	gd = {
		i: digitize(g.take(i, axis), n_bins)
		for i in range(shape[0])
	}
	zd = {
		j: digitize(z.take(j, axis), n_bins)
		for j in range(shape[1])
	}
	if parallel:
		looper = itertools.product(
			range(shape[0]), range(shape[1]))
		with joblib.parallel_backend('multiprocessing'):
			mi_normalized = joblib.Parallel(n_jobs=n_jobs)(
				joblib.delayed(_mi)(gd[i], zd[j]) for i, j in looper
			)
		mi_normalized = np.reshape(mi_normalized, shape)
	else:
		mi_normalized = np.zeros(shape)
		for i in range(shape[0]):
			for j in range(shape[1]):
				mi_normalized[i, j] = _mi(gd[i], zd[j])
	return mi_normalized


def _mi(g, z):
	mi_gz = sk_metric.mutual_info_score(g, z)
	ent_g = sk_metric.mutual_info_score(g, g)
	return mi_gz / ent_g


def digitize(a: np.ndarray, n_bins: int):
	bins = np.histogram(a, bins=n_bins)[1]
	inds = np.digitize(a, bins[:-1], False)
	return inds


def entropy_discrete(a: np.ndarray, n_bins: int):
	a = digitize(a, n_bins)
	ent = sk_metric.mutual_info_score(a, a)
	ent /= np.log(n_bins)
	return ent


def entropy_normalized(p: np.ndarray, axis: int):
	return sp_stats.entropy(p, axis=axis, base=p.shape[axis])


def flatten_stim(x):
	return flatten_arr(x, ndim_end=0, ndim_start=1)


def null_adj_ll(
		true: np.ndarray,
		pred: np.ndarray,
		axis: int = 0,
		normalize: bool = True,
		return_lls: bool = False, ):
	kws = {
		'true': true,
		'axis': axis,
		'normalize': normalize,
	}
	ll = poisson_ll(pred=pred, **kws)
	null = poisson_ll(pred=true.mean(axis), **kws)
	if return_lls:
		return ll, null
	else:
		return ll - null


def poisson_ll(
		true: np.ndarray,
		pred: np.ndarray,
		axis: int = 0,
		normalize: bool = True,
		eps: float = 1e-8, ):
	ll = np.sum(true * np.log(pred + eps) - pred, axis=axis)
	if normalize:
		ll /= np.maximum(eps, np.sum(true, axis=axis))
	return ll


def skew(x: np.ndarray, axis: int = 0):
	x1 = np.expand_dims(np.expand_dims(np.take(
		x, 0, axis=axis), axis=axis), axis=axis)
	x2 = np.expand_dims(np.expand_dims(np.take(
		x, 1, axis=axis), axis=axis), axis=axis)
	x3 = np.expand_dims(np.expand_dims(np.take(
		x, 2, axis=axis), axis=axis), axis=axis)
	s1 = np.concatenate([np.zeros_like(x1), -x3, x2], axis=axis+1)
	s2 = np.concatenate([x3, np.zeros_like(x2), -x1], axis=axis+1)
	s3 = np.concatenate([-x2, x1, np.zeros_like(x3)], axis=axis+1)
	s = np.concatenate([s1, s2, s3], axis=axis)
	return s


def radself2polar(
		a: np.ndarray,
		b: np.ndarray,
		dtype=float, ):
	ta = np.tan(a, dtype=dtype)
	tb = np.tan(b, dtype=dtype)
	theta = np.sqrt(ta**2 + tb**2)
	theta = np.arctan(theta, dtype=dtype)
	phi = np.arctan2(tb, ta, dtype=dtype)
	phi[phi < 0] += 2 * np.pi
	return theta, phi


def vel2polar(
		a: np.ndarray,
		axis: int = None,
		eps: float = 1e-10, ):
	if axis is None:
		s = a.shape
		if collections.Counter(s)[2] != 1:
			msg = f"provide axis, bad shape:\n{s}"
			raise RuntimeError(msg)
		axis = next(
			i for i, d in
			enumerate(s)
			if d == 2
		)
	vx = np.take(a, 0, axis=axis)
	vy = np.take(a, 1, axis=axis)
	rho = sp_lin.norm(a, axis=axis)
	phi = np.arccos(vx / np.maximum(rho, eps))
	phi[vy < 0] = 2 * np.pi - phi[vy < 0]
	phi[rho == 0] = np.nan
	return rho, phi


def cart2polar(x: np.ndarray):
	x, shape = _check_input(x)
	r = sp_lin.norm(x, ord=2, axis=-1)
	theta = np.arccos(x[:, 2] / r)
	phi = np.arctan2(x[:, 1], x[:, 0])
	phi[phi < 0] += 2 * np.pi
	out = np.concatenate([
		np.expand_dims(r, -1),
		np.expand_dims(theta, -1),
		np.expand_dims(phi, -1),
	], axis=-1)
	if len(shape) > 2:
		out = out.reshape(shape)
	return out


def polar2cart(r: np.ndarray):
	r, shape = _check_input(r)
	x = r[:, 0] * np.sin(r[:, 1]) * np.cos(r[:, 2])
	y = r[:, 0] * np.sin(r[:, 1]) * np.sin(r[:, 2])
	z = r[:, 0] * np.cos(r[:, 1])
	out = np.concatenate([
		np.expand_dims(x, -1),
		np.expand_dims(y, -1),
		np.expand_dims(z, -1),
	], axis=-1)
	if len(shape) > 2:
		out = out.reshape(shape)
	return out


class Obj(object):
	def __init__(self, verbose: bool = False):
		super(Obj, self).__init__()
		self.verbose = verbose
		self.datetime = now(True)

	def setattrs(self, **attrs):
		for k, v in attrs.items():
			setattr(self, k, v)
		return


def _check_input(e: np.ndarray):
	if not isinstance(e, np.ndarray):
		e = np.array(e)
	shape = e.shape
	if e.ndim == 1:
		assert len(e) == 3
		e = e.reshape(-1, 3)
	elif e.ndim == 2:
		assert e.shape[1] == 3
	else:
		e = flatten_arr(e)
	return e, shape
