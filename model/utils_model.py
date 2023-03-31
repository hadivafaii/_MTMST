from utils.plotting import *
from .configuration import ConfigVAE, ConfigTrain
from torch.nn import functional as F
from torch import nn


def beta_anneal_cosine(
		n_iters: int,
		start: float = 0.0,
		stop: float = 1.0,
		n_cycles: int = 4,
		portion: float = 0.5,
		beta: float = 1.0, ):
	period = n_iters / n_cycles
	step = (stop-start) / (period*portion)
	betas = np.ones(n_iters) * beta
	for c in range(n_cycles):
		v, i = start, 0
		while v <= stop:
			val = (1 - np.cos(v*np.pi)) * beta / 2
			betas[int(i+c*period)] = val
			v += step
			i += 1
	return betas


def beta_anneal_linear(
		n_iters: int,
		beta: float = 1,
		anneal_portion: float = 0.3,
		constant_portion: float = 0,
		min_beta: float = 1e-4, ):
	betas = np.ones(n_iters) * beta
	a = int(np.ceil(constant_portion * n_iters))
	b = int(np.ceil((constant_portion + anneal_portion) * n_iters))
	betas[:a] = min_beta
	betas[a:b] = np.linspace(min_beta, beta, b - a)
	return betas


def kl_balancer(
		kl_all: List[torch.Tensor],
		alpha: torch.Tensor = None,
		coeff: float = 1.0,
		beta: float = 1.0,
		eps: float = 0.01, ):
	kl_all = torch.stack(kl_all, dim=1)
	kl_vals = torch.mean(kl_all, dim=0)
	if alpha is not None and coeff < beta:
		gamma = torch.mean(
			kl_all.detach().abs(),
			keepdim=True,
			dim=0,
		) + eps
		gamma *= alpha.unsqueeze(0)
		gamma /= torch.mean(
			gamma, keepdim=True, dim=1)
		kl = torch.sum(kl_all * gamma, dim=1)
		gamma = gamma.squeeze(0)
	else:
		kl = torch.sum(kl_all, dim=1)
		gamma = torch.ones(len(kl_vals))
	return kl.mul(coeff), gamma, kl_vals


def kl_balancer_coeff(groups: List[int], fun: str):
	n = len(groups)
	if fun == 'equal':
		coeff = torch.cat([
			torch.ones(groups[n-i-1])
			for i in range(n)
		], dim=0)
	elif fun == 'linear':
		coeff = torch.cat([
			(2 ** i) * torch.ones(groups[n-i-1])
			for i in range(n)
		], dim=0)
	elif fun == 'sqrt':
		coeff = torch.cat([
			np.sqrt(2 ** i) * torch.ones(groups[n-i-1])
			for i in range(n)
		], dim=0)
	elif fun == 'square':
		coeff = torch.cat([
			np.square(2 ** i) / groups[n-i-1] * torch.ones(groups[n-i-1])
			for i in range(n)], dim=0)
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
		if tot == 0:
			continue
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
		model_name: str,
		fit_name: Union[str, int] = -1,
		chkpt: int = -1,
		device: str = 'cpu',
		strict: bool = True,
		verbose: bool = False,
		path: str = 'Documents/MTMST/models', ):
	# cfg model
	path = pjoin(os.environ['HOME'], path, model_name)
	fname = next(s for s in os.listdir(path) if 'json' in s)
	with open(pjoin(path, fname), 'r') as f:
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
	if isinstance(fit_name, str):
		path = pjoin(path, fit_name)
	elif isinstance(fit_name, int):
		path = sorted(filter(
			os.path.isdir, [
				pjoin(path, e) for e
				in os.listdir(path)
			]
		), key=_sort_fn)[fit_name]
	else:
		raise ValueError(fit_name)
	files = sorted(os.listdir(path))
	# state dict
	fname_pt = [
		f for f in files if
		f.split('.')[-1] == 'pt'
	][chkpt]
	state_dict = pjoin(path, fname_pt)
	state_dict = torch.load(state_dict)
	ema = state_dict['model_ema'] is not None
	model.load_state_dict(
		state_dict=state_dict['model'],
		strict=strict,
	)
	# cfg train
	fname = next(
		f for f in files if
		f.split('.')[-1] == 'json'
	)
	with open(pjoin(path, fname), 'r') as f:
		cfg_train = json.load(f)
	cfg_train = ConfigTrain(**cfg_train)
	fname = fname.split('.')[0]
	fname = fname.replace('Config', '')
	if fname == 'Train':
		from .train_vae import TrainerVAE
		trainer = TrainerVAE(
			model=model,
			cfg=cfg_train,
			device=device,
			verbose=verbose,
			ema=ema,
		)
	else:
		raise NotImplementedError
	if ema:
		trainer.model_ema.load_state_dict(
			state_dict=state_dict['model_ema'],
			strict=strict,
		)
	trainer.optim.load_state_dict(
		state_dict['optim'])
	trainer.scaler.load_state_dict(
		state_dict['scaler'])
	if trainer.optim_schedule is not None:
		trainer.optim_schedule.load_state_dict(
			state_dict.get('scheduler', {}))
	metadata = {
		**state_dict['metadata'],
		'file': fname_pt,
		'path': path,
	}
	return trainer, metadata


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

	def save(
			self,
			checkpoint: int = -1,
			name: str = None,
			path: str = None, ):
		path = path if path else self.chkpt_dir
		name = name if name else type(self).__name__
		fname = '-'.join([
			name,
			f"{checkpoint:04d}",
			f"({now(True)}).bin",
		])
		fname = pjoin(path, fname)
		torch.save(self.state_dict(), fname)
		return fname


def _sort_fn(f: str):
	f = f.split('(')[-1].split(')')[0]
	ymd, hm = f.split(',')
	yy, mm, dd = ymd.split('_')
	h, m = hm.split(':')
	yy, mm, dd, h, m = map(
		lambda s: int(s),
		[yy, mm, dd, h, m],
	)
	x = (
		yy * 1e8 +
		mm * 1e6 +
		dd * 1e4 +
		h * 1e2 +
		m
	)
	return x
