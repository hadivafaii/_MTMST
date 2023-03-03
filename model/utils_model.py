from utils.plotting import *
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from .configuration import ConfigVAE, ConfigTrain


def kl_coeff(step, total_step, constant_step, min_kl_coeff):
	return max(min((step - constant_step) / total_step, 1.0), min_kl_coeff)


def kl_balancer(kl_all, coeff=1.0, alpha=None):
	if alpha is not None and coeff < 1.0:
		alpha = alpha.unsqueeze(0)

		kl_all = torch.stack(kl_all, dim=1)
		kl_coeff_i, kl_vals = kl_per_group(kl_all)
		total_kl = torch.sum(kl_coeff_i)

		kl_coeff_i = kl_coeff_i / alpha * total_kl
		kl_coeff_i = kl_coeff_i / torch.mean(kl_coeff_i, dim=1, keepdim=True)
		kl = torch.sum(kl_all * kl_coeff_i.detach(), dim=1)

		# for reporting
		kl_coeffs = kl_coeff_i.squeeze(0)
	else:
		kl_all = torch.stack(kl_all, dim=1)
		kl_vals = torch.mean(kl_all, dim=0)
		kl = torch.sum(kl_all, dim=1)
		kl_coeffs = torch.ones(size=(len(kl_vals),))

	return coeff * kl, kl_coeffs, kl_vals


def kl_per_group(kl_all):
	kl_vals = torch.mean(kl_all, dim=0)
	kl_coeff_i = torch.abs(kl_all)
	kl_coeff_i = torch.mean(kl_coeff_i, dim=0, keepdim=True) + 0.01

	return kl_coeff_i, kl_vals


def kl_balancer_coeff(groups: List[int], fun: str, device: torch.device = None):
	n = len(groups)
	if fun == 'equal':
		coeff = torch.cat([
			torch.ones(groups[n-i-1])
			for i in range(n)
		], dim=0).to(device)
	elif fun == 'linear':
		coeff = torch.cat([
			(2 ** i) * torch.ones(groups[n-i-1])
			for i in range(n)
		], dim=0).to(device)
	elif fun == 'sqrt':
		coeff = torch.cat([
			np.sqrt(2 ** i) * torch.ones(groups[n-i-1])
			for i in range(n)
		], dim=0).to(device)
	elif fun == 'square':
		coeff = torch.cat([
			np.square(2 ** i) / groups[n-i-1] * torch.ones(groups[n-i-1])
			for i in range(n)], dim=0).to(device)
	else:
		raise NotImplementedError(fun)
	# convert min to 1.
	coeff /= torch.min(coeff)
	return coeff


class AvgrageMeter(object):
	def __init__(self):
		self.avg = 0
		self.sum = 0
		self.cnt = 0
		self.reset()

	def reset(self):
		self.avg = 0
		self.sum = 0
		self.cnt = 0

	def update(self, val, n=1):
		self.sum += val * n
		self.cnt += n
		self.avg = self.sum / self.cnt


def print_num_params(module: nn.Module):
	t = PrettyTable(['Module Name', 'Num Params'])
	for name, m in module.named_modules():
		tot = sum(
			p.numel() for p
			in m.parameters()
			if p.requires_grad
		)
		if tot // 1e6 > 0:
			tot = f"{np.round(tot / 1e6, 2):1.1f} Mil"
		elif tot // 1e3 > 0:
			tot = f"{np.round(tot / 1e3, 2):1.1f} K"

		if '.' not in name:
			if isinstance(m, type(module)):
				t.add_row([m.__class__.__name__, tot])
				t.add_row(['---', '---'])
			else:
				t.add_row([name, tot])
	print(t, '\n\n')
	return


def load_model(
		name: str,
		fit: int = -1,
		chkpt: int = -1,
		strict: bool = True,
		verbose: bool = False,
		load_dir: str = 'Documents/MTMST/models', ):
	# cfg model
	load_dir = pjoin(os.environ['HOME'], load_dir, name)
	fname = next(s for s in os.listdir(load_dir) if 'json' in s)
	with open(pjoin(load_dir, fname), 'r') as f:
		cfg = json.load(f)
	cfg = ConfigVAE(**cfg)
	fname = fname.split('.')[0]
	fname = fname.replace('Config', '')
	if fname == 'VAE':
		from .vae2d import VAE
		model = VAE(cfg, verbose=verbose)
	else:
		raise NotImplementedError
	# now enter the fit folder
	load_dir = sorted(filter(
		os.path.isdir, [
			pjoin(load_dir, e) for e
			in os.listdir(load_dir)
		]
	))[fit]
	files = sorted(os.listdir(load_dir))
	# cfg train
	fname = next(
		f for f in files if
		f.split('.')[-1] == 'json'
	)
	with open(pjoin(load_dir, fname), 'r') as f:
		cfg_train = json.load(f)
	cfg_train = ConfigTrain(**cfg_train)
	# bin
	fname = [
		f for f in files if
		f.split('.')[-1] == 'bin'
	][chkpt]
	state_dict = pjoin(load_dir, fname)
	state_dict = torch.load(state_dict)
	model.load_state_dict(
		state_dict=state_dict,
		strict=strict,
	)
	model.eval()
	meta = {
		'fname': fname,
		'chkpt': chkpt,
		'dir': load_dir,
	}
	return model, cfg_train, meta


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


def add_weight_decay(
		model: nn.Module,
		weight_decay: float = 1e-2,
		skip: Tuple[str, ...] = ('bias',), ):
	decay = []
	no_decay = []
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		if len(param.shape) <= 1 or any(k in name for k in skip):
			no_decay.append(param)
		else:
			decay.append(param)
	param_groups = [
		{'params': no_decay, 'weight_decay': 0.},
		{'params': decay, 'weight_decay': weight_decay},
	]
	return param_groups


class Module(nn.Module):
	def __init__(self, cfg, verbose: bool = False):
		super(Module, self).__init__()
		self.cfg = cfg
		self.datetime = now(True)
		self.verbose = verbose
		self.chkpt_dir = None

	def print(self):
		print_num_params(self)

	def create_chkpt_dir(self, comment: str = None):
		chkpt_dir = pjoin(
			self.cfg.save_dir,
			'_'.join([
				comment if comment else
				f"seed-{self.cfg.seed}",
				f"({self.datetime})",
			]),
		)
		os.makedirs(chkpt_dir, exist_ok=True)
		self.chkpt_dir = chkpt_dir
		return

	def save(self, checkpoint: int = -1, path: str = None):
		path = path if path else self.chkpt_dir
		fname = '-'.join([
			type(self).__name__,
			f"{checkpoint:04d}",
			f"({now(True)}).bin",
		])
		fname = pjoin(path, fname)
		torch.save(self.state_dict(), fname)
		return fname


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


def overlap_score(m1: np.ndarray, m2: np.ndarray):
	numer = np.logical_and(m1.astype(bool), m2.astype(bool)).sum()
	denum = min(m1.astype(bool).sum(), m2.astype(bool).sum())
	return numer / denum


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


def vel2polar(a: np.ndarray, eps: float = 1e-10):
	vx, vy = a[..., 0], a[..., 1]
	rho = sp_lin.norm(a, ord=2, axis=-1)
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


def self2polar(
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
