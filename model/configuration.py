from utils.generic import *
from utils.process import load_cellinfo


_CONV_NORM_CHOICES = ['weight', 'spectral', None]
_SCHEDULER_CHOICES = ['cosine', 'exponential', 'step', 'cyclic', None]
_OPTIM_CHOICES = ['adamw', 'adamax', 'sgd']


class BaseConfig(object):
	def __init__(
			self,
			name: str,
			seed: int = 0,
			full: bool = False,
			h_file: str = 'MTLFP_tres25',
			h_pre: str = 'simulation_dim-19_1e+05',
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
			self.h_pre = pjoin(self.data_dir, f"{h_pre}.h5")
			self._mkdirs()
			self._load_cellinfo()
			self.seed = seed
			self.set_seed()

	def get_all_dirs(self):
		dirs = {k: getattr(self, k) for k in dir(self) if '_dir' in k}
		dirs = filter(lambda x: isinstance(x[1], str), dirs.items())
		return dict(dirs)

	def _mkdirs(self):
		for _dir in self.get_all_dirs().values():
			os.makedirs(_dir, exist_ok=True)

	def _load_cellinfo(self):
		self.useful_cells = load_cellinfo(
			pjoin(self.base_dir, 'extra_info'))

	def set_seed(self):
		torch.manual_seed(self.seed)
		torch.cuda.manual_seed(self.seed)
		torch.cuda.manual_seed_all(self.seed)
		os.environ["SEED"] = str(self.seed)
		np.random.seed(self.seed)
		random.seed(self.seed)
		return

	def save(self, save_dir: str, verbose: bool = False):
		fname = type(self).__name__
		file = pjoin(save_dir, f"{fname}.json")
		if os.path.isfile(file):
			return
		params = inspect.signature(self.__init__).parameters
		vals = {
			k: getattr(self, k) for k, p in params.items()
			if int(p.kind) == 1 and hasattr(self, k)
			# first 'if' gets rid of args, kwargs
		}
		save_obj(
			obj=vals,
			file_name=fname,
			save_dir=save_dir,
			verbose=verbose,
			mode='json',
		)
		return

	def get_name(self):
		raise NotImplementedError


class ConfigVAE(BaseConfig):
	def __init__(
			self,
			n_kers: int = 4,
			n_rots: int = 8,
			ker_sz: int = 4,
			input_sz: int = 19,
			n_pre_cells: int = 3,
			n_pre_blocks: int = 1,
			n_post_cells: int = 3,
			n_post_blocks: int = 1,
			n_latent_scales: int = 2,
			n_groups_per_scale: int = 4,
			n_latent_per_group: int = 2,
			n_cells_per_cond: int = 2,
			n_power_iter: int = 5,
			activation_fn: str = 'swish',
			spectral_reg: bool = False,
			weight_norm: bool = True,
			residual_kl: bool = True,
			rot_equiv: bool = True,
			ada_groups: bool = True,
			compress: bool = True,
			use_bn: bool = False,
			use_se: bool = True,
			create: bool = True,
			**kwargs,
	):
		self.n_kers = n_kers
		self.n_rots = n_rots
		self.ker_sz = ker_sz
		self.input_sz = input_sz
		self.n_pre_cells = n_pre_cells
		self.n_pre_blocks = n_pre_blocks
		self.n_post_cells = n_post_cells
		self.n_post_blocks = n_post_blocks
		self.n_latent_scales = n_latent_scales
		self.n_groups_per_scale = n_groups_per_scale
		self.n_latent_per_group = n_latent_per_group
		self.n_cells_per_cond = n_cells_per_cond
		self.spectral_reg = spectral_reg
		self.rot_equiv = rot_equiv
		self.compress = compress
		self.use_bn = use_bn
		self.groups = groups_per_scale(
			n_scales=self.n_latent_scales,
			n_groups_per_scale=self.n_groups_per_scale,
			is_adaptive=ada_groups,
		)
		super(ConfigVAE, self).__init__(
			name=self.get_name(),
			full=True,
			**kwargs,
		)
		self.weight_norm = weight_norm
		self.activation_fn = activation_fn
		self.n_power_iter = n_power_iter
		self.residual_kl = residual_kl
		self.ada_groups = ada_groups
		self.use_se = use_se
		if create:
			self.save(self.save_dir)

	def total_latents(self):
		return sum(self.groups) * self.n_latent_per_group

	def get_name(self):
		name = [
			'x'.join([
				f"k-{self.n_kers}",
				str(self.n_rots),
			]).replace(' ', ''),
			'x'.join([
				f"z-{self.n_latent_per_group}",
				str(self.groups),
			]).replace(' ', ''),
			'x'.join([
				f"pre-{self.n_pre_blocks}",
				str(self.n_pre_cells),
			]).replace(' ', ''),
			'x'.join([
				f"post-{self.n_post_blocks}",
				str(self.n_post_cells),
			]).replace(' ', ''),
		]
		name = '_'.join(name)
		if self.rot_equiv:
			name = f"{name}_rot"
		if self.compress:
			name = f"{name}_cmprs"
		if self.use_bn:
			name = f"{name}_bn"
		return name


class ConfigTrain(BaseConfig):
	def __init__(
			self,
			lr: float = 1e-1,
			epochs: int = 400,
			batch_size: int = 64,
			warmup_epochs: int = 30,
			optimizer: str = 'adamax',
			optimizer_kws: dict = None,
			lambda_init: float = 10,
			lambda_norm: float = 1e-2,
			lambda_anneal: bool = False,
			balanced_recon: bool = True,
			kl_const_coeff: float = 1e-4,
			kl_const_portion: float = 1e-4,
			kl_anneal_portion: float = 0.3,
			scheduler_type: str = 'cosine',
			scheduler_kws: dict = None,
			clip_grad: float = None,
			chkpt_freq: int = 10,
			eval_freq: int = 5,
			log_freq: int = 2,
			**kwargs,
	):
		super(ConfigTrain, self).__init__(
			name=self.get_name(),
			full=False,
			**kwargs,
		)
		self.lr = lr
		self.epochs = epochs
		self.batch_size = batch_size
		self.warmup_epochs = warmup_epochs
		if lambda_anneal:
			assert lambda_init > 0 and lambda_norm > 0
		self.lambda_init = lambda_init
		self.lambda_norm = lambda_norm
		self.lambda_anneal = lambda_anneal
		self.kl_const_coeff = kl_const_coeff
		self.kl_const_portion = kl_const_portion
		self.kl_anneal_portion = kl_anneal_portion
		assert optimizer in _OPTIM_CHOICES,\
			f"allowed optimizers:\n{_OPTIM_CHOICES}"
		self.optimizer = optimizer
		self._set_optim_kws(optimizer_kws)
		assert scheduler_type in _SCHEDULER_CHOICES,\
			f"allowed schedulers:\n{_SCHEDULER_CHOICES}"
		self.scheduler_type = scheduler_type
		self._set_scheduler_kws(scheduler_kws)
		self.balanced_recon = balanced_recon
		self.clip_grad = clip_grad
		self.chkpt_freq = chkpt_freq
		self.eval_freq = eval_freq
		self.log_freq = log_freq

	def get_name(self):
		return 'TrainerVAE'
	# TODO: create this function,
	#  will go to comment in train()

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
		lr_min = 1e-4
		period = float(
			self.epochs - 1 -
			self.warmup_epochs
		)
		if self.scheduler_type == 'cosine':
			defaults = {
				'T_max': period,
				'eta_min': lr_min,
			}
		elif self.scheduler_type == 'exponential':
			defaults = {
				'gamma': 0.9,
				'eta_min': 1e-4,
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


def groups_per_scale(
		n_scales: int,
		n_groups_per_scale: int,
		is_adaptive: bool = True,
		min_groups: int = 1,
		divider: int = 2, ):
	assert n_groups_per_scale >= 1
	n = n_groups_per_scale
	g = []
	for s in range(n_scales):
		g.append(n)
		if is_adaptive:
			n = n // divider
			n = max(min_groups, n)
	return g
