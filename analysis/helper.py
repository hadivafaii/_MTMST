from utils.plotting import *
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


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
		normalize: bool = True, ):
	eps = 1e-5
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
	def __init__(
			self,
			name: str = None,
			tres: int = 25,
			verbose: bool = False,
	):
		super(Obj, self).__init__()
		self.name = name
		self.tres = tres
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
