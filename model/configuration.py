from utils.generic import *
from utils.process import load_cellinfo


_NORM_CHOICES = ['batch', 'layer', 'group']
_CONV_NORM_CHOICES = ['weight', 'spectral']


class BaseConfig(object):
	def __init__(
			self,
			name: str,
			seed: int = 42,
			makedirs: bool = True,
			init_range: float = 0.01,
			h_file: str = 'MTLFP_tres25.h5',
			base_dir: str = 'Documents/MTMST/MTLFP',
	):
		super(BaseConfig, self).__init__()

		self.base_dir = pjoin(os.environ['HOME'], base_dir)
		self.results_dir = pjoin(self.base_dir, 'results')
		self.fit_dir = pjoin(self.base_dir, 'fits', name)
		self.save_dir = pjoin(self.base_dir, 'models', name)
		self.h_file = pjoin(self.base_dir, 'xtracted_python', h_file)

		if makedirs:
			self._mkdirs()

		self.init_range = init_range
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
			z_dim: int = 8,
			n_lvls: int = 2,
			n_kers: int = 4,
			n_rots: int = 8,
			ker_sz: int = 3,
			input_sz: int = 16,
			n_pre_cells: int = 3,
			n_pre_blocks: int = 2,
			n_latent_scales: int = 2,
			n_groups_per_scale: int = 10,
			n_latent_per_group: int = 20,
			n_cells_per_cond: int = 2,
			norm: str = 'batch',
			conv_norm: str = 'spectral',
			activation_fn: str = 'relu',
			upsample_mode: str = 'linear',
			residual_kl: bool = True,
			use_dilation: bool = False,
			use_bias: bool = False,
			use_se: bool = True,
			**kwargs,
	):
		assert norm in _NORM_CHOICES,\
			f"allowed normalizations:\n{_NORM_CHOICES}"
		assert conv_norm in _CONV_NORM_CHOICES,\
			f"allowed normalizations:\n{_CONV_NORM_CHOICES}"

		self.z_dim = z_dim
		self.n_lvls = n_lvls
		self.n_kers = n_kers
		self.n_rots = n_rots
		self.ker_sz = ker_sz
		self.input_sz = input_sz
		self.n_pre_cells = n_pre_cells
		self.n_pre_blocks = n_pre_blocks
		self.n_latent_scales = n_latent_scales
		self.n_groups_per_scale = n_groups_per_scale
		self.n_latent_per_group = n_latent_per_group
		self.n_cells_per_cond = n_cells_per_cond
		self.norm = norm
		self.conv_norm = conv_norm
		super(ConfigVAE, self).__init__(self.get_name(), **kwargs)

		self.groups = groups_per_scale(
			n_scales=self.n_latent_scales,
			n_groups_per_scale=self.n_groups_per_scale,
		)
		self.activation_fn = activation_fn
		self.upsample_mode = upsample_mode
		self.residual_kl = residual_kl
		self.use_dilation = use_dilation
		self.use_bias = use_bias
		self.use_se = use_se
		self.save(self.save_dir)

	def get_name(self):
		return '_'.join([
			f"z-{self.z_dim}x{self.n_lvls}",
			f"k-{self.n_kers}x{self.n_rots}",
			f"norm-{self.norm}-{self.conv_norm}",
		])


def groups_per_scale(
		n_scales: int,
		n_groups_per_scale: int,
		is_adaptive: bool = True,
		min_groups: int = 1,
		divider: int = 2, ):
	g = []
	n = n_groups_per_scale
	for s in range(n_scales):
		assert n >= 1
		g.append(n)
		if is_adaptive:
			n = n // divider
			n = max(min_groups, n)
	return g
