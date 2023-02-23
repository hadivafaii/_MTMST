from utils.generic import *
from utils.process import load_cellinfo


_NORM_CHOICES = ['batch', 'layer', 'group']
_CONV_NORM_CHOICES = ['weight', 'spectral']
_SCHEDULER_CHOICES = ['cosine', 'exponential', 'step', 'cyclic', None]
_OPTIM_CHOICES = ['adamw', 'adamax', 'sgd']


class BaseConfig(object):
	def __init__(
			self,
			name: str,
			seed: int = 0,
			makedirs: bool = True,
			# init_range: float = 0.01,
			h_file: str = 'MTLFP_tres25.h5',
			base_dir: str = 'Documents/MTMST',
	):
		super(BaseConfig, self).__init__()

		self.base_dir = pjoin(os.environ['HOME'], base_dir)
		self.results_dir = pjoin(self.base_dir, 'results')
		self.runs_dir = pjoin(self.base_dir, 'runs', name)
		self.save_dir = pjoin(self.base_dir, 'models', name)
		self.h_file = pjoin(self.base_dir, 'xtracted_python', h_file)

		if makedirs:
			self._mkdirs()

		# self.init_range = init_range
		self._load_cellinfo()
		self.seed = seed
		self.set_seed()

	def _get_all_dirs(self):
		dirs = {k: getattr(self, k) for k in dir(self) if '_dir' in k}
		dirs = filter(lambda x: isinstance(x[1], str), dirs.items())
		return dict(dirs)

	def _mkdirs(self):
		for _dir in self._get_all_dirs().values():
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
			k: getattr(self, k)
			for k, p in params.items()
			if int(p.kind) == 1  # gets rid of args, kwargs
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
			scale_mult: int = 2,
			n_pre_cells: int = 3,
			n_pre_blocks: int = 1,
			n_post_cells: int = 3,
			n_post_blocks: int = 1,
			n_latent_scales: int = 2,
			n_groups_per_scale: int = 4,
			n_latent_per_group: int = 20,
			n_cells_per_cond: int = 2,
			# norm: str = 'batch',
			conv_norm: str = 'spectral',
			activation_fn: str = 'swish',
			# upsample_mode: str = 'bilinear',
			residual_kl: bool = True,
			# use_dilation: bool = False,
			# use_bias: bool = False,
			ada_groups: bool = False,
			use_bn: bool = False,
			use_se: bool = True,
			**kwargs,
	):
		# assert norm in _NORM_CHOICES,\
		# 	f"allowed normalizations:\n{_NORM_CHOICES}"
		assert conv_norm in _CONV_NORM_CHOICES,\
			f"allowed normalizations:\n{_CONV_NORM_CHOICES}"

		self.n_kers = n_kers
		self.n_rots = n_rots
		self.ker_sz = ker_sz
		self.input_sz = input_sz
		self.scale_mult = scale_mult
		self.n_pre_cells = n_pre_cells
		self.n_pre_blocks = n_pre_blocks
		self.n_post_cells = n_post_cells
		self.n_post_blocks = n_post_blocks
		self.n_latent_scales = n_latent_scales
		self.n_groups_per_scale = n_groups_per_scale
		self.n_latent_per_group = n_latent_per_group
		self.n_cells_per_cond = n_cells_per_cond
		# self.norm = norm
		self.conv_norm = conv_norm
		super(ConfigVAE, self).__init__(
			self.get_name(), **kwargs)

		self.groups = groups_per_scale(
			n_scales=self.n_latent_scales,
			n_groups_per_scale=self.n_groups_per_scale,
			is_adaptive=ada_groups,
		)
		self.activation_fn = activation_fn
		# self.upsample_mode = upsample_mode
		self.residual_kl = residual_kl
		# self.use_dilation = use_dilation
		# self.use_bias = use_bias
		self.ada_groups = ada_groups
		self.use_bn = use_bn
		self.use_se = use_se
		self.save(self.save_dir)

	def get_name(self):
		return '_'.join([
			# f"groups-{self.groups}",
			f"k-{self.n_kers}x{self.n_rots}",
			f"norm-{self.conv_norm}",
			# TODO: this needs revision, add things such as
			#  self.groups
		])


class ConfigTrain(BaseConfig):
	def __init__(
			self,
			lr: float = 1e-2,
			beta1: float = 0.9,
			beta2: float = 0.999,
			batch_size: int = 64,
			optimizer: str = 'adamw',
			weight_decay: float = 1e-2,
			beta_warmup_steps: int = None,
			scheduler_type: str = 'cosine',
			scheduler_gamma: float = 0.9,
			scheduler_period: int = 10,
			lr_min: float = 1e-8,
			log_freq: int = 100,
			chkpt_freq: int = 1,
			eval_freq: int = 5,
			xv_folds: int = 5,
			**kwargs,
	):
		self.lr = lr
		self.beta1 = beta1
		self.beta2 = beta2
		self.batch_size = batch_size
		self.weight_decay = weight_decay
		self.beta_warmup_steps = beta_warmup_steps

		assert optimizer in _OPTIM_CHOICES,\
			f"allowed optimizers:\n{_OPTIM_CHOICES}"
		self.optimizer = optimizer
		assert scheduler_type in _SCHEDULER_CHOICES,\
			f"allowed schedulers:\n{_SCHEDULER_CHOICES}"
		self.scheduler_type = scheduler_type
		self.scheduler_gamma = scheduler_gamma
		self.scheduler_period = scheduler_period
		self.lr_min = lr_min

		self.log_freq = log_freq
		self.chkpt_freq = chkpt_freq
		self.eval_freq = eval_freq
		self.xv_folds = xv_folds

		super(ConfigTrain, self).__init__(
			self.get_name(), **kwargs)

		self.save(self.save_dir)

	def get_name(self):
		raise NotImplementedError


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
