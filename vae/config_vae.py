from base.config_base import *


class ConfigVAE(BaseConfig):
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
		self.groups = _groups_per_scale(
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


class ConfigTrainVAE(BaseConfigTrain):
	def __init__(
			self,
			kl_beta: float = 1.0,
			kl_beta_min: float = 1e-4,
			kl_balancer: str = 'equal',
			kl_anneal_cycles: int = 1,
			kl_anneal_portion: float = 0.3,
			kl_const_portion: float = 1e-4,
			lambda_anneal: bool = True,
			lambda_init: float = 1e-7,
			lambda_norm: float = 1e-4,
			spectral_reg: bool = False,
			**kwargs,
	):
		defaults = dict(
			lr=0.002,
			epochs=2000,
			batch_size=500,
			warmup_portion=0.025,
			optimizer='adamax_fast',
			optimizer_kws=None,
			scheduler_type='cosine',
			scheduler_kws=None,
			ema_rate=0.999,
			grad_clip=1000,
			use_amp=False,
			chkpt_freq=50,
			eval_freq=10,
			log_freq=20,
		)
		kwargs = setup_kwargs(defaults, kwargs)
		super(ConfigTrainVAE, self).__init__(**kwargs)
		self.kl_beta = kl_beta
		self.kl_beta_min = kl_beta_min
		self.kl_balancer = kl_balancer
		self.kl_anneal_cycles = kl_anneal_cycles
		self.kl_anneal_portion = kl_anneal_portion
		self.kl_const_portion = kl_const_portion
		self.lambda_anneal = lambda_anneal
		self.lambda_init = lambda_init
		self.lambda_norm = lambda_norm
		self.spectral_reg = spectral_reg

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


def _groups_per_scale(
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
