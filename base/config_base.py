from utils.generic import *
from utils.process import load_cellinfo
_SCHEDULER_CHOICES = ['cosine', 'exponential', 'step', 'cyclic', None]
_OPTIM_CHOICES = ['adamax', 'adam', 'adamw', 'radam', 'sgd', 'adamax_fast']


class BaseConfig(object):
	def __init__(
			self,
			name: str,
			seed: int = 0,
			save: bool = True,
			full: bool = False,
			h_file: str = 'ALL_tres25',
			sim_path: str = 'fixate1_dim-65_n-750k',
			base_dir: str = 'Documents/MTMST',
	):
		super(BaseConfig, self).__init__()
		if full:
			self.base_dir = pjoin(os.environ['HOME'], base_dir)
			self.results_dir = pjoin(self.base_dir, 'results')
			self.runs_dir = pjoin(self.base_dir, 'runs', name)
			self.save_dir = pjoin(self.base_dir, 'models', name)
			self.data_dir = pjoin(self.base_dir, 'data')
			self.h_file = pjoin(self.data_dir, f"{h_file}.h5")
			self.sim_path = pjoin(self.data_dir, sim_path)
			self._load_cellinfo()
			self.seed = seed
			self.set_seed()
		if save:
			self._mkdirs()
			self.save()

	def name(self):
		raise NotImplementedError

	def save(self, save_dir: str = None, verbose: bool = False):
		save_dir = save_dir if save_dir else self.save_dir
		_save_config(self, save_dir, False, verbose)

	def get_all_dirs(self):
		dirs = {k: getattr(self, k) for k in dir(self) if '_dir' in k}
		dirs = filter(lambda x: isinstance(x[1], str), dirs.items())
		return dict(dirs)

	def _mkdirs(self):
		for _dir in self.get_all_dirs().values():
			os.makedirs(_dir, exist_ok=True)

	def _load_cellinfo(self):
		useful = load_cellinfo(
			pjoin(self.base_dir, 'extra_info'))
		with h5py.File(self.h_file) as file:
			for expt in file['YUWEI']:
				if expt in useful:
					continue
				useful[expt] = [0]
		useful = dict(sorted(useful.items()))
		self.useful_yuwei = useful
		return

	def set_seed(self):
		torch.manual_seed(self.seed)
		torch.cuda.manual_seed(self.seed)
		torch.cuda.manual_seed_all(self.seed)
		os.environ["SEED"] = str(self.seed)
		np.random.seed(self.seed)
		random.seed(self.seed)
		return


class BaseConfigTrain(object):
	def __init__(
			self,
			lr: float,
			epochs: int,
			batch_size: int,
			warm_restart: int,
			warmup_portion: float,
			optimizer: str,
			optimizer_kws: dict,
			scheduler_type: str,
			scheduler_kws: dict,
			ema_rate: float = 0.999,
			grad_clip: float = 1000,
			use_amp: bool = False,
			chkpt_freq: int = 50,
			eval_freq: int = 10,
			log_freq: int = 20,
	):
		super(BaseConfigTrain, self).__init__()
		self.lr = lr
		self.epochs = epochs
		self.batch_size = batch_size
		assert warm_restart >= 0
		assert warmup_portion >= 0
		self.warm_restart = warm_restart
		self.warmup_portion = warmup_portion
		assert optimizer in _OPTIM_CHOICES, \
			f"allowed optimizers:\n{_OPTIM_CHOICES}"
		self.optimizer = optimizer
		self._set_optim_kws(optimizer_kws)
		assert scheduler_type in _SCHEDULER_CHOICES, \
			f"allowed schedulers:\n{_SCHEDULER_CHOICES}"
		self.scheduler_type = scheduler_type
		self._set_scheduler_kws(scheduler_kws)
		self.ema_rate = ema_rate
		self.grad_clip = grad_clip
		self.chkpt_freq = chkpt_freq
		self.eval_freq = eval_freq
		self.log_freq = log_freq
		self.use_amp = use_amp

	def name(self):
		raise NotImplementedError

	def save(self, save_dir: str, verbose: bool = False):
		_save_config(self, save_dir, True, verbose)

	def _set_optim_kws(self, kws):
		defaults = {
			'betas': (0.9, 0.999),
			'weight_decay': 3e-4,
			'eps': 1e-8,
		}
		kws = setup_kwargs(defaults, kws)
		self.optimizer_kws = kws
		return

	def _set_scheduler_kws(self, kws):
		lr_min = 1e-5
		period = self.epochs * (1 - self.warmup_portion)
		period /= (2 * self.warm_restart + 1)
		if self.scheduler_type == 'cosine':
			defaults = {
				'T_max': period,
				'eta_min': lr_min,
			}
		elif self.scheduler_type == 'exponential':
			defaults = {
				'gamma': 0.9,
				'eta_min': lr_min,
			}
		elif self.scheduler_type == 'step':
			defaults = {
				'gamma': 0.1,
				'step_size': 10,
			}
		elif self.scheduler_type == 'cyclic':
			defaults = {
				'max_lr': self.lr,
				'base_lr': lr_min,
				'mode': 'exp_range',
				'step_size_up': period,
				'step_size': 10,
				'gamma': 0.9,
			}
		else:
			raise NotImplementedError(self.scheduler_type)
		kws = setup_kwargs(defaults, kws)
		self.scheduler_kws = kws
		return


def _save_config(
		obj,
		save_dir: str,
		with_base: bool = True,
		verbose: bool = False, ):
	fname = type(obj).__name__
	file = pjoin(save_dir, f"{fname}.json")
	if os.path.isfile(file):
		return
	params = inspect.signature(obj.__init__).parameters
	if with_base:
		base = type(obj).__bases__[0]
		base = inspect.signature(base.__init__).parameters
		params = {**params, **base}
	vals = {
		k: getattr(obj, k) for
		k, p in params.items()
		if _param_checker(k, p, obj)
	}
	save_obj(
		obj=vals,
		file_name=fname,
		save_dir=save_dir,
		verbose=verbose,
		mode='json',
	)
	return


def _param_checker(k, p, obj):
	# second 'if' gets rid of args, kwargs
	return k != 'self' and int(p.kind) == 1 and hasattr(obj, k)
