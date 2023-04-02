from utils.generic import *
from utils.process import load_cellinfo
_SCHEDULER_CHOICES = ['cosine', 'exponential', 'step', 'cyclic', None]
_OPTIM_CHOICES = ['adamax', 'adam', 'adamw', 'radam', 'sgd', 'adamax_fast']


class _BaseConfig(object):
	def __init__(
			self,
			name: str = 'Base',
			seed: int = 0,
			full: bool = False,
			save: bool = False,
			h_file: str = 'MTLFP_tres25',
			h_pre: str = 'simulation_dim-19_5e+04',
			base_dir: str = 'Documents/MTMST',
	):
		super(_BaseConfig, self).__init__()
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
		if save:
			self.save()

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

	def save(self, save_dir: str = None, verbose: bool = False):
		fname = type(self).__name__
		save_dir = save_dir if save_dir else self.save_dir
		file = pjoin(save_dir, f"{fname}.json")
		if os.path.isfile(file):
			return
		params = inspect.signature(self.__init__).parameters
		vals = {
			k: getattr(self, k) for k, p in params.items()
			if (int(p.kind) == 1 and hasattr(self, k))
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

	def name(self):
		raise NotImplementedError


class ConfigVAE(_BaseConfig):
	def __init__(
			self,
			n_kers: int = 4,
			n_rots: int = 8,
			ker_sz: int = 4,
			input_sz: int = 19,
			n_enc_cells: int = 2,
			n_enc_nodes: int = 2,
			n_dec_cells: int = 2,
			n_dec_nodes: int = 1,
			n_pre_cells: int = 3,
			n_pre_blocks: int = 1,
			n_post_cells: int = 3,
			n_post_blocks: int = 1,
			n_latent_scales: int = 3,
			n_groups_per_scale: int = 4,
			n_latent_per_group: int = 5,
			activation_fn: str = 'swish',
			balanced_recon: bool = True,
			residual_kl: bool = True,
			scale_init: bool = False,
			rot_equiv: bool = False,
			ada_groups: bool = True,
			spectral_norm: int = 0,
			separable: bool = False,
			compress: bool = True,
			use_bn: bool = False,
			use_se: bool = True,
			full: bool = True,
			**kwargs,
	):
		self.n_kers = n_kers
		self.n_rots = n_rots
		self.ker_sz = ker_sz
		self.input_sz = input_sz
		self.n_enc_cells = n_enc_cells
		self.n_enc_nodes = n_enc_nodes
		self.n_dec_cells = n_dec_cells
		self.n_dec_nodes = n_dec_nodes
		self.n_pre_cells = n_pre_cells
		self.n_pre_blocks = n_pre_blocks
		self.n_post_cells = n_post_cells
		self.n_post_blocks = n_post_blocks
		self.n_latent_scales = n_latent_scales
		self.n_groups_per_scale = n_groups_per_scale
		self.n_latent_per_group = n_latent_per_group
		self.spectral_norm = spectral_norm
		self.rot_equiv = rot_equiv
		self.separable = separable
		self.compress = compress
		self.use_bn = use_bn
		self.groups = groups_per_scale(
			n_scales=self.n_latent_scales,
			n_groups_per_scale=self.n_groups_per_scale,
			is_adaptive=ada_groups,
		)
		super(ConfigVAE, self).__init__(
			name=self.name(),
			full=full,
			**kwargs,
		)
		self.balanced_recon = balanced_recon
		self.activation_fn = activation_fn
		self.residual_kl = residual_kl
		self.scale_init = scale_init
		self.ada_groups = ada_groups
		self.use_se = use_se

	def total_latents(self):
		return sum(self.groups) * self.n_latent_per_group

	def name(self):
		name = [
			'x'.join([
				f"k-{self.n_kers}",
				str(self.n_rots),
			]).replace(' ', '') if self.rot_equiv
			else f"k-{self.n_kers * self.n_rots}",
			'x'.join([
				f"z-{self.n_latent_per_group}",
				str(list(reversed(self.groups))),
			]).replace(' ', ''),
			'-'.join([
				'x'.join([
					f"enc({self.n_enc_cells}",
					f"{self.n_enc_nodes})",
				]).replace(' ', ''),
				'x'.join([
					f"dec({self.n_dec_cells}",
					f"{self.n_dec_nodes})",
				]).replace(' ', ''),
				'x'.join([
					f"pre({self.n_pre_blocks}",
					f"{self.n_pre_cells})",
				]).replace(' ', ''),
				'x'.join([
					f"post({self.n_post_blocks}",
					f"{self.n_post_cells})",
				]).replace(' ', ''),
			]),
		]
		name = '_'.join(name)
		if self.spectral_norm:
			name = f"{name}_sn-{self.spectral_norm}"
		if self.rot_equiv:
			name = f"{name}_rot"
		if self.separable:
			name = f"{name}_sep"
		if not self.compress:
			name = f"{name}_noncmprs"
		if self.use_bn:
			name = f"{name}_bn"
		return name


class ConfigTrain(_BaseConfig):
	def __init__(
			self,
			lr: float = 0.001,
			epochs: int = 1000,
			batch_size: int = 512,
			warmup_portion: float = 0.02,
			optimizer: str = 'adamax_fast',
			optimizer_kws: dict = None,
			lambda_anneal: bool = True,
			lambda_init: float = 1e-7,
			lambda_norm: float = 1e-3,
			kl_beta: float = 1.0,
			kl_beta_min: float = 1e-4,
			kl_anneal_cycles: int = 1,
			kl_anneal_portion: float = 0.3,
			kl_const_portion: float = 1e-3,
			kl_balancer: str = 'equal',
			scheduler_type: str = 'cosine',
			scheduler_kws: dict = None,
			spectral_reg: bool = False,
			ema_rate: float = 1 - 1e-3,
			grad_clip: float = 1000,
			use_amp: bool = False,
			chkpt_freq: int = 50,
			eval_freq: int = 5,
			log_freq: int = 20,
	):
		super(ConfigTrain, self).__init__(
			full=False, save=False)
		self.lr = lr
		self.epochs = epochs
		self.batch_size = batch_size
		self.warmup_portion = warmup_portion
		self.lambda_anneal = lambda_anneal
		self.lambda_init = lambda_init
		self.lambda_norm = lambda_norm
		self.kl_beta = kl_beta
		self.kl_beta_min = kl_beta_min
		self.kl_balancer = kl_balancer
		self.kl_anneal_cycles = kl_anneal_cycles
		self.kl_anneal_portion = kl_anneal_portion
		self.kl_const_portion = kl_const_portion
		assert optimizer in _OPTIM_CHOICES,\
			f"allowed optimizers:\n{_OPTIM_CHOICES}"
		self.optimizer = optimizer
		self._set_optim_kws(optimizer_kws)
		assert scheduler_type in _SCHEDULER_CHOICES,\
			f"allowed schedulers:\n{_SCHEDULER_CHOICES}"
		self.scheduler_type = scheduler_type
		self._set_scheduler_kws(scheduler_kws)
		self.spectral_reg = spectral_reg
		self.ema_rate = ema_rate
		self.grad_clip = grad_clip
		self.chkpt_freq = chkpt_freq
		self.eval_freq = eval_freq
		self.log_freq = log_freq
		self.use_amp = use_amp

	def name(self):
		name = [
			'-'.join([
				f"ep{self.epochs}",
				f"b{self.batch_size}",
				f"lr({self.lr:0.2g})"]),
			'-'.join([
				f"beta({self.kl_beta:0.2g}"
				f":{self.kl_anneal_cycles}"
				f"x{self.kl_anneal_portion:0.1f})",
			])
		]
		if self.lambda_norm > 0:
			name.append(f"lamb({self.lambda_norm:0.2g})")
		if self.grad_clip is not None:
			name.append(f"gr({self.grad_clip})")
		return '_'.join(name)

	def _set_optim_kws(self, kws):
		defaults = {
			'betas': (0.9, 0.999),
			'weight_decay': 1e-4,
			'eps': 1e-8,
		}
		kws = setup_kwargs(defaults, kws)
		self.optimizer_kws = kws
		return

	def _set_scheduler_kws(self, kws):
		lr_min = 1e-5
		period = np.round(
			self.epochs *
			(1 - self.warmup_portion)
		) - 1
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


def groups_per_scale(
		n_scales: int,
		n_groups_per_scale: int,
		is_adaptive: bool = True,
		min_groups: int = 1,
		divider: int = 2, ):
	assert n_groups_per_scale >= 1
	n = n_groups_per_scale
	g = []
	for _ in range(n_scales):
		g.append(n)
		if is_adaptive:
			n = n // divider
			n = max(min_groups, n)
	return g
