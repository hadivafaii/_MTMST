from base.config_base import *
_POOL_CHOICES = ['none', 'max', 'avg']
_ACT_CHOICES = ['none', 'softplus', 'learned_softplus']


class ConfigReadout(BaseConfig):
	def __init__(
			self,
			n_ftrs: Dict[int, int],
			n_lags: int = 21,
			n_tkers: int = 1,
			n_skers: int = 1,
			n_cells: int = 1,
			dropout: float = 0.0,
			act_fn: str = 'none',
			pool: str = 'max',
			full: bool = True,
			**kwargs,
	):
		self.n_ftrs = n_ftrs
		self.n_lags = n_lags
		self.n_tkers = n_tkers
		self.n_skers = n_skers
		self.n_cells = n_cells
		self.dropout = dropout
		assert act_fn in _ACT_CHOICES,\
			f"allowed act fn:\n{_ACT_CHOICES}"
		self.act_fn = act_fn
		assert pool in _POOL_CHOICES,\
			f"allowed pooling:\n{_POOL_CHOICES}"
		self.pool = pool
		super(ConfigReadout, self).__init__(
			name=self.name(),
			full=full,
			save=False,  # TODO: remvoe later
			**kwargs,
		)

	def name(self):
		return 'test'


class ConfigTrainReadout(BaseConfigTrain):
	def __init__(
			self,
			kl_beta: float = 1.0,
			**kwargs,
	):
		defaults = dict(
			lr=0.01,
			epochs=500,
			batch_size=1000,
			warmup_portion=0.025,
			optimizer='adamw',
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
		super(ConfigTrainReadout, self).__init__(**kwargs)
		self.kl_beta = kl_beta

	def name(self):
		raise NotImplementedError
