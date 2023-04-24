from .helper import *
from base.dataset import load_ephys
from vae.train_vae import TrainerVAE
from .linear import compute_sta, LinearModel
from base.common import (
	load_model_lite, load_model,
	get_act_fn, nn, F,
)

_ATTRS = [
	'root', 'expt', 'n_pcs', 'n_lags', 'n_top_pix', 'rescale',
	'use_latents', 'kws_hf', 'kws_push', 'kws_xt', 'normalize',
]
_FIT = [
	'sta', 'temporal', 'spatial', 'top_lags', 'top_pix_per_lag',
	'sorted_pix', 'has_repeats', 'nc', 'max_perf', 'mu', 'sd',
	'glm', 'pca', 'mod', 'best_pix', 'best_lag', 'perf', 'df',
]


class Readout(object):
	def __init__(
			self,
			root: str,
			expt: str,
			tr: TrainerVAE = None,
			n_lags: int = 12,
			n_pcs: int = 500,
			n_top_pix: int = 4,
			rescale: float = 2.0,
			dtype: str = 'float32',
			normalize: bool = False,
			verbose: bool = False,
			**kwargs,
	):
		super(Readout, self).__init__()
		self.tr = tr
		self.root = root
		self.expt = expt
		self.n_pcs = n_pcs
		self.n_lags = n_lags
		self.rescale = rescale
		self.kws_hf = {
			k: kwargs[k] if k in kwargs else v for k, v in
			dict(dim=17, sres=1, apply_mask=True).items()
		}
		self.kws_xt = {
			k: kwargs[k] if k in kwargs else v for k, v in
			dict(scale=2, pool='avg', act_fn='none').items()
		}
		self.kws_push = {
			k: kwargs[k] if k in kwargs else v for k, v
			in dict(which='z', use_ema=False).items()
		}
		self.use_latents = self.kws_push['which'] == 'z'
		self.n_top_pix = min(n_top_pix, self.kws_xt['scale'] ** 2)
		self.normalize = normalize
		self.verbose = verbose
		self.dtype = dtype
		self.glm = None
		# neuron attributes
		self.max_perf = None
		self.stim, self.stim_r = None, None
		self.spks, self.spks_r = None, None
		self.good, self.good_r = None, None
		self.has_repeats, self.nc = None, None
		# fitted attributes
		self.logger = None
		self.best_pix = {}
		self.best_lag = {}
		self.pca, self.mod = {}, {}
		self.perf, self.df = {}, {}

	def fit_readout(
			self,
			path: str = None,
			zscore: bool = True, ):
		if path is not None:
			self.logger = make_logger(
				path=path,
				name=type(self).__name__,
				level=logging.WARNING,
			)
		self.load_neurons()
		self._xtract()
		self._sta(zscore)
		self._top_lags()
		self._top_pix()
		return self

	def fit_neuron(
			self,
			idx: int = 0,
			glm: bool = False,
			lags: List[int] = None,
			alphas: List[float] = None,
			**kwargs, ):

		def _update(_r, _best_r):
			if self.has_repeats:
				perf_r[inds] = linmod.df['r_tst'].max()
				perf_r2[inds] = linmod.df['r2_tst'].max()
			else:
				perf_r[inds] = _r
			if _r > _best_r:
				self.perf[idx] = _r
				self.df[idx] = linmod.df
				self.mod[idx] = linmod.models[a]
				self.best_pix[idx] = pix
				self.best_lag[idx] = lag
				self.pca[idx] = pc
			return

		self.glm = glm
		if self.glm:
			kws_model = dict(
				category='PoissonRegressor',
				alphas=alphas if alphas else
				np.logspace(-7, 1, num=9),
			)
		else:
			kws_model = dict(
				category='Ridge',
				alphas=alphas if alphas else
				np.logspace(-4, 8, num=13),
			)
		if lags is None:
			lags = [self.top_lags[idx]]
		assert isinstance(lags, Collection)

		if self.use_latents:
			shape = (len(lags),)
			looper = enumerate(lags)
		else:
			shape = (len(lags), *self.spatial.shape[1:])
			looper = itertools.product(
				enumerate(lags),
				self.sorted_pix[idx],
			)
		best_a = None
		best_r = -np.inf
		perf_r = np.zeros(shape)
		perf_r2 = np.zeros(shape)
		for item in looper:
			if self.use_latents:
				pix = None
				lag_i, lag = item
				inds = lag_i
			else:
				(lag_i, lag), pix = item
				inds = (lag_i, *pix)
			data = self.get_data(idx, pix=pix, lag=lag)
			if not self.use_latents:
				pc = sk_decomp.PCA(
					n_components=self.n_pcs,
					svd_solver='full',
				)
				data['x'] = pc.fit_transform(data['x'])
				if self.has_repeats:
					data['x_tst'] = pc.transform(data['x_tst'])
			else:
				pc = None
			kws_model.update(data)
			linmod = LinearModel(**kws_model)
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				linmod.fit_linear(**kwargs)
			_ = linmod.best_alpha()
			# Fit linear regression
			kws_lr = kws_model.copy()
			kws_lr['category'] = 'LinearRegression'
			lr = LinearModel(**kws_lr)
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				lr.fit_linear(**kwargs)
			_ = lr.best_alpha()
			# merge together
			linmod.df = pd.concat([linmod.df, lr.df])
			linmod.models.update(lr.models)
			linmod.preds.update(lr.preds)
			a, r = linmod.best_alpha()
			_update(r, best_r)
			if r > best_r:
				best_r = r
				best_a = a

			if self.verbose:
				msg = '-' * 80
				msg += f"\n{self.expt}, "
				msg += f"neuron # {idx}; "
				msg += f"inds: (lag, \[i, j]) = ({lag}, "
				if self.use_latents:
					msg += ')'
				else:
					msg += f"{pix[0]}, {pix[1]})"
				print(msg)
				print(linmod.df)
				linmod.show_pred()
				print('~' * 80)
				print('\n')

		if self.verbose:
			msg = f"{self.expt}, "
			msg += f"neuron # {idx};  "
			msg += f"best alpha = {best_a:0.2g}, "
			msg += f"best_r = {best_r:0.3f}"
			print(msg)

		return perf_r, perf_r2

	def validate(self, idx: int):
		kws = dict(
			tr=self.tr,
			kws_process=self.kws_xt,
			verbose=self.verbose,
			dtype=self.dtype,
			max_pool=False,
			**self.kws_push,
		)
		ftr, _ = push(stim=self.stim, **kws)
		ftr_r, _ = push(stim=self.stim_r, **kws)
		# global normalzie
		ftr = normalize_global(ftr, self.mu, self.sd)
		ftr_r = normalize_global(ftr_r, self.mu, self.sd)
		data = self.get_data(idx, ftr, ftr_r)
		if not self.use_latents:
			data['x'] = self.pca[idx].transform(data['x'])
			if self.has_repeats:
				data['x_tst'] = self.pca[idx].transform(data['x_tst'])
		return data

	def forward(
			self,
			x: np.ndarray = None,
			full: bool = False, ):
		if x is None:
			x = self.stim
		kws = dict(
			tr=self.tr,
			kws_process=self.kws_xt,
			verbose=self.verbose,
			dtype=self.dtype,
			max_pool=True,
			**self.kws_push,
		)
		ftr, ftr_p = push(stim=x, **kws)
		if full:
			dims = (0, -2, -1)
			var = np.var(ftr, axis=dims)
			mu2 = np.mean(ftr, axis=dims) ** 2
			stats = dict(
				var=var,
				mu2=mu2,
				snr2=mu2 / var,
				s=sp_lin.svdvals(ftr_p),
			)
		else:
			stats = {}
		ftr = normalize_global(ftr, self.mu, self.sd)
		return ftr, ftr_p, stats

	def get_data(
			self,
			idx: int,
			ftr: np.ndarray = None,
			ftr_r: np.ndarray = None,
			pix: Tuple[int, int] = None,
			lag: int = None, ):
		if ftr is None:
			ftr = self.ftr
		if ftr_r is None:
			ftr_r = self.ftr_r
		if lag is None:
			lag = self.best_lag[idx]
		if pix is None and not self.use_latents:
			pix = self.best_pix[idx]
		kws = dict(
			lag=lag,
			x=ftr if self.use_latents
			else ftr[..., pix[0], pix[1]],
			y=self.spks[:, idx],
			good=self.good,
		)
		if self.has_repeats:
			kws.update(dict(
				x_tst=ftr_r if self.use_latents
				else ftr_r[..., pix[0], pix[1]],
				y_tst=self.spks_r[idx],
				good_tst=self.good_r,
			))
		return setup_data(**kws)

	def load_neurons(self):
		if self.nc is not None:
			return self
		f = h5py.File(self.tr.model.cfg.h_file)
		g = f[self.root][self.expt]
		self.has_repeats = g.attrs.get('has_repeats')
		stim, spks, mask, stim_r, spks_r, good_r = load_ephys(
			group=g,
			kws_hf=self.kws_hf,
			rescale=self.rescale,
			dtype=self.dtype,
		)
		if self.has_repeats:
			self.max_perf = max_r2(spks_r)
		self.good, self.good_r = np.where(mask)[0], good_r
		self.stim, self.stim_r = stim, stim_r
		self.spks, self.spks_r = spks, spks_r
		self.nc = spks.shape[1]
		f.close()

		if self.verbose:
			print('[PROGRESS] neural data loaded')
		return self

	def load(self, fit_name: str, device: str, glm: bool = False):
		self.glm = glm
		path = _setup_path(fit_name, glm)
		# load pickle
		file = f"{self.name()}.pkl"
		file = pjoin(path, file)
		with (open(file, 'rb')) as f:
			pkl = pickle.load(f)
		for k, v in pkl.items():
			setattr(self, k, v)
		# load Trainer
		if self.tr is None:
			path = pjoin(path, 'Trainer')
			self.tr, _ = load_model_lite(
				path=path,
				device=device,
				verbose=self.verbose,
			)
		return self

	def save(self, path: str):
		path_tr = pjoin(path, 'Trainer')
		os.makedirs(path_tr, exist_ok=True)

		# save trainer
		cond = any(
			f for f in os.listdir(path_tr)
			if f.endswith('.pt')
		)
		if not cond:
			self.tr.save(path_tr)
			self.tr.cfg.save(path_tr)
			self.tr.model.cfg.save(path_tr)

		# save pickle
		save_obj(
			obj=self.state_dict(),
			file_name=self.name(),
			save_dir=path,
			mode='pkl',
			verbose=self.verbose,
		)
		return

	def state_dict(self):
		return {k: getattr(self, k) for k in _ATTRS + _FIT}

	def name(self):
		return f"{self.root}-{self.expt}"

	def show(self, idx: int = 0):
		fig, axes = create_figure(
			1, 2, (8.0, 2.5),
			width_ratios=[3, 1],
			constrained_layout=True,
		)
		axes[0].plot(self.temporal[idx], marker='o')
		axes[0].axvline(
			self.n_lags - self.top_lags[idx],
			color='r',
			ls='--',
			label=f'best lag = {self.top_lags[idx]}',
		)
		axes[0].set_xlabel('Lag [ms]', fontsize=12)
		xticklabels = [
			f"-{(self.n_lags - i) * 25}"
			for i in range(self.n_lags + 1)
		]
		axes[0].set(
			xticks=range(0, self.n_lags + 1),
			xticklabels=xticklabels,
		)
		axes[0].tick_params(axis='x', rotation=0, labelsize=9)
		axes[0].tick_params(axis='y', labelsize=9)
		axes[0].legend(fontsize=11)
		axes[0].grid()

		if self.spatial is None:
			axes[1].remove()
		else:
			sns.heatmap(
				data=self.spatial[idx],
				annot_kws={'fontsize': 10},
				cmap='rocket',
				square=True,
				cbar=False,
				annot=True,
				fmt='1.3g',
				ax=axes[1],
			)
		plt.show()
		return

	def _xtract(self):
		kws = dict(
			tr=self.tr,
			kws_process=self.kws_xt,
			verbose=self.verbose,
			dtype=self.dtype,
			max_pool=False,
			**self.kws_push,
		)
		ftr, _ = push(stim=self.stim, **kws)
		ftr_r, _ = push(stim=self.stim_r, **kws)
		# normalize?
		self.mu = ftr.mean() if self.normalize else 0
		self.sd = ftr.std() if self.normalize else 1
		self.ftr = normalize_global(ftr, self.mu, self.sd)
		self.ftr_r = normalize_global(ftr_r, self.mu, self.sd)
		if self.verbose:
			print('[PROGRESS] features extracted')
		return

	def _sta(self, zscore: bool = False):
		self.sta = compute_sta(
			stim=self.ftr,
			good=self.good,
			spks=self.spks,
			n_lags=self.n_lags,
			verbose=self.verbose,
			zscore=zscore,
		)
		if self.verbose:
			print('[PROGRESS] sta computed')
		return

	def _top_lags(self):
		axis = 2 if self.use_latents else (2, 3, 4)
		self.temporal = np.mean(self.sta ** 2, axis=axis)
		self.top_lags = np.argmax(self.temporal[:, ::-1], axis=1)
		if self.verbose:
			print('[PROGRESS] best lag estimated')
		return

	def _top_pix(self):
		if self.kws_push['which'] == 'z':
			self.top_pix_per_lag = None
			self.sorted_pix = None
			self.spatial = None
			return
		# top pix per lag
		shape = (self.nc, self.n_lags + 1, 2)
		self.top_pix_per_lag = np.zeros(shape, dtype=int)
		looper = itertools.product(
			range(self.nc),
			range(self.n_lags + 1),
		)
		for idx, lag in looper:
			t = self.n_lags - lag
			norm = self.sta[idx][t]
			norm = np.mean(norm ** 2, axis=0)
			i, j = np.unravel_index(
				np.argmax(norm), norm.shape)
			self.top_pix_per_lag[idx, t] = i, j
		# top pix overall
		self.spatial = np.zeros((self.nc, *self.sta.shape[-2:]))
		for idx in range(self.nc):
			norm = self.sta[idx] ** 2
			norm = np.mean(norm, axis=(0, 1))
			self.spatial[idx] = norm
		self.sorted_pix = np.zeros((self.nc, self.n_top_pix, 2), dtype=int)
		for idx in range(self.nc):
			top = np.array(list(zip(*np.unravel_index(np.argsort(
				self.spatial[idx].ravel()), self.spatial.shape[1:]))))
			self.sorted_pix[idx] = top[::-1][:self.n_top_pix]
		return


def summarize_readout_fits(
		fit_name: str,
		device: str = 'cpu',
		glm: bool = False, ):
	path = _setup_path(fit_name, glm)
	args = pjoin(path, 'args.json')
	with open(args, 'r') as f:
		args = json.load(f)
	tr = pjoin(path, 'Trainer')
	tr, _ = load_model_lite(
		tr, device, strict=False)

	df, df_all, ro_all = [], [], {}
	for f in sorted(os.listdir(path)):
		if not f.endswith('.pkl'):
			continue
		root = f.split('.')[0]
		root, expt = root.split('-')
		kws = dict(tr=tr, root=root, expt=expt)
		ro = Readout(**kws).load(fit_name, 'cpu')
		ro_all[f"{root}_{expt}"] = ro
		for i, d in ro.df.items():
			_df = d.reset_index()
			if ro.max_perf is not None:
				_df['max_r2'] = ro.max_perf[i]
			_df['root'] = root
			_df['expt'] = expt
			_df['cell'] = i
			df_all.append(_df)
		# alpha & perf
		log_alpha = {}
		for i, m in ro.mod.items():
			if hasattr(m, 'alpha'):
				log_alpha[i] = np.log10(m.alpha)
			else:
				log_alpha[i] = -10  # For lr: alpha = 0
		if ro.max_perf is not None:
			perf = {
				i: r / np.sqrt(ro.max_perf[i])
				for i, r in ro.perf.items()
			}
		else:
			perf = ro.perf
		# put all in df
		df.append({
			'root': [root] * len(perf),
			'expt': [expt] * len(perf),
			'cell': perf.keys(),
			'perf': perf.values(),
			'log_alpha': log_alpha.values(),
			'best_lag': ro.best_lag.values(),
			'top_lag': ro.top_lags[list(perf.keys())],
		})
		# pixel stuff
		if not ro.use_latents:
			pix_ranks, pix_counts = {}, {}
			for i, best in ro.best_pix.items():
				pix_ranks[i] = np.where(np.all(
					ro.sorted_pix[i] == best,
					axis=1
				))[0][0]
				pix_counts[i] = collections.Counter([
					tuple(e) for e in
					ro.top_pix_per_lag[i]
				]).get(best, 0)
			df[-1].update({
				'pix_rank': pix_ranks.values(),
				'pix_count': pix_counts.values(),
			})
	df = pd.DataFrame(merge_dicts(df))
	df_all = pd.concat(df_all).reset_index()
	save_obj(
		obj=df,
		file_name='summary',
		save_dir=path,
		verbose=False,
		mode='df',
	)
	save_obj(
		obj=df_all,
		file_name='summary_all',
		save_dir=path,
		verbose=False,
		mode='df',
	)
	return df, df_all, ro_all, args, tr


def push(
		tr: TrainerVAE,
		stim: np.ndarray,
		kws_process: dict,
		which: str = 'enc',
		verbose: bool = False,
		use_ema: bool = False,
		max_pool: bool = False,
		dtype: str = 'float32', ):
	if stim is None:
		return None, None
	# feature sizes
	assert kws_process['pool'] != 'none'
	s = kws_process['scale']
	m = tr.select_model(use_ema)
	nf_enc, nf_dec = m.ftr_sizes()
	if which == 'enc':
		nf = sum(nf_enc.values())
		shape = (len(stim), nf, s, s)
	elif which == 'dec':
		nf = sum(nf_dec.values())
		shape = (len(stim), nf, s, s)
	elif which == 'z':
		nf = m.cfg.total_latents()
		shape = (len(stim), nf)
	else:
		raise NotImplementedError(which)

	x = np.empty(shape, dtype=dtype)
	if max_pool:
		assert which in ['enc', 'dec']
		shape = (len(stim), nf)
		xp = np.empty(shape, dtype=dtype)
		mp = nn.AdaptiveMaxPool2d(1)
	else:
		xp, mp = None, None
	n_iter = len(x) / tr.cfg.batch_size
	n_iter = int(np.ceil(n_iter))
	for i in tqdm(range(n_iter), disable=not verbose):
		a = i * tr.cfg.batch_size
		b = min(a + tr.cfg.batch_size, len(x))
		_, z, *_, ftr = m.xtract_ftr(
			tr.to(stim[a:b]), full=True)
		if which == 'z':
			z = torch.cat(z, dim=1).squeeze()
			x[a:b] = to_np(z).astype(dtype)
		else:
			ftr = process_ftrs(ftr[which], **kws_process)
			x[a:b] = to_np(ftr).astype(dtype)
			if max_pool:
				xp[a:b] = to_np(mp(F.silu(ftr)).squeeze())
	return x, xp


def process_ftrs(
		ftr: dict,
		scale: int = 4,
		pool: str = 'max',
		act_fn: str = 'swish', ):
	# activation
	activation = get_act_fn(act_fn)
	if activation is not None:
		ftr = {
			s: activation(x) for
			s, x in ftr.items()
		}
	# pool
	if pool == 'max':
		pool = nn.AdaptiveMaxPool2d(scale)
	elif pool == 'avg':
		pool = nn.AdaptiveAvgPool2d(scale)
	else:
		raise NotImplementedError
	for s, x in ftr.items():
		if s != scale:
			ftr[s] = pool(x)
	ftr = torch.cat(list(ftr.values()), dim=1)
	return ftr


def setup_data(
		lag: int,
		x: np.ndarray,
		y: np.ndarray,
		good: np.ndarray,
		x_tst: np.ndarray = None,
		y_tst: np.ndarray = None,
		good_tst: np.ndarray = None, ):
	inds = good.copy()
	inds = inds[inds > lag]
	data = dict(x=x[inds - lag], y=y[inds])
	if x_tst is not None:
		data.update({
			'x_tst': x_tst[good_tst - lag],
			'y_tst': np.nanmean(y_tst, 0),
		})
	return data


def _setup_path(fit_name: str, glm: bool = False):
	path = 'Documents/MTMST/results'
	path = pjoin(
		pjoin(os.environ['HOME'], path),
		'GLM' if glm else 'Ridge',
		fit_name,
	)
	return path


def _setup_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"model_name",
		help='which VAE to load',
		type=str,
	)
	parser.add_argument(
		"fit_name",
		help='which VAE fit to load',
		type=str,
	)
	parser.add_argument(
		"device",
		help='cuda:n',
		type=str,
	)
	parser.add_argument(
		"--root",
		help="choices: {'YUWEI', 'NARDIN', 'CRCNS'}",
		default='YUWEI',
		type=str,
	)
	parser.add_argument(
		"--checkpoint",
		help='checkpoint',
		default=-1,
		type=int,
	)
	parser.add_argument(
		"--reservoir",
		help='revert back to untrained?',
		action='store_true',
		default=False,
	)
	parser.add_argument(
		"--comment",
		help='added to fit name',
		default=None,
		type=str,
	)
	# Readout related
	parser.add_argument(
		"--n_pcs",
		help='# PC components',
		default=500,
		type=int,
	)
	parser.add_argument(
		"--n_lags",
		help='# time lags',
		default=12,
		type=int,
	)
	parser.add_argument(
		"--n_top_pix",
		help='# top pixels to loop over',
		default=4,
		type=int,
	)
	parser.add_argument(
		"--rescale",
		help='HyperFlow stim rescale',
		default=2.0,
		type=float,
	)
	parser.add_argument(
		"--apply_mask",
		help='HyperFlow: apply mask or full field?',
		default=True,
		type=bool,
	)
	parser.add_argument(
		'--log_alphas',
		help='List of log alpha values',
		default=None,
		type=float,
		nargs='+',
	)
	parser.add_argument(
		"--which",
		help="which to use: {'enc', 'dec', 'z'}",
		default='z',
		type=str,
	)
	parser.add_argument(
		"--scale",
		help='Which scale to pool to?',
		default=2,
		type=int,
	)
	parser.add_argument(
		"--pool",
		help="choices: {'max', 'avg'}",
		default='avg',
		type=str,
	)
	parser.add_argument(
		"--act_fn",
		help="choices: {'swish', 'none'}",
		default='none',
		type=str,
	)
	parser.add_argument(
		"--use_ema",
		help='use ema or main model?',
		default=False,
		type=bool,
	)
	parser.add_argument(
		"--glm",
		help='GLM or Ridge?',
		default=False,
		type=bool,
	)
	parser.add_argument(
		"--zscore",
		help='zscore stim before STA?',
		default=True,
		type=bool,
	)
	parser.add_argument(
		"--normalize",
		help='normalize before PCA?',
		action='store_true',
		default=False,
	)
	# etc.
	parser.add_argument(
		"--verbose",
		help='verbose?',
		default=False,
		type=bool,
	)
	parser.add_argument(
		"--dry_run",
		help='to make sure config is alright',
		action='store_true',
		default=False,
	)
	return parser.parse_args()


def _main():
	args = _setup_args()
	# setup alphas
	if args.log_alphas is None:
		log_a = itertools.chain(
			range(-8, 3, 2),
			range(3, 7),
			range(8, 17, 2),
		)
		args.log_alphas = sorted(log_a)

	# load trainer
	tr, metadata = load_model(
		model_name=args.model_name,
		fit_name=args.fit_name,
		device=args.device,
		checkpoint=args.checkpoint,
		strict=False,  # TODO: later remove this
	)
	# reservoir?
	if args.reservoir:
		tr.reset_model()
		name = 'reservoir'
		args.checkpoint = 0
	else:
		name = tr.model.cfg.sim
		args.checkpoint = metadata['checkpoint']

	# print args
	print(args)

	# create save path
	if args.comment is not None:
		name = f"{args.comment}_{name}"
	nf = sum(tr.model.ftr_sizes()[0].values())
	fit_name = '_'.join([
		name,
		f"nf-{nf}",
		f"({now(True)})",
	])
	path = pjoin(
		tr.model.cfg.results_dir,
		'GLM' if args.glm else 'Ridge',
		fit_name,
	)
	# save args
	if not args.dry_run:
		os.makedirs(path, exist_ok=True)
		save_obj(
			obj=vars(args),
			file_name='args',
			save_dir=path,
			mode='json',
			verbose=args.verbose,
		)
	print(f"\nname: {fit_name}\n")

	kws = dict(
		tr=tr,
		root=args.root,
		n_pcs=args.n_pcs,
		n_lags=args.n_lags,
		n_top_pix=args.n_top_pix,
		rescale=args.rescale,
		normalize=args.normalize,
		verbose=args.verbose,
		which=args.which,
		apply_mask=args.apply_mask,
		use_ema=args.use_ema,
		act_fn=args.act_fn,
		scale=args.scale,
		pool=args.pool,
	)
	kws_fit = dict(
		glm=args.glm,
		alphas=[
			10 ** a for a in
			args.log_alphas],
		lags=range(args.n_lags + 1),
	)
	if not args.dry_run:
		pbar = tqdm(
			tr.model.cfg.useful_yuwei.items(),
			dynamic_ncols=True,
			leave=True,
			position=0,
		)
		for expt, useful in pbar:
			ro = Readout(expt=expt, **kws).fit_readout(
				path=path, zscore=args.zscore)
			for idx in useful:
				_ = ro.fit_neuron(idx=idx, **kws_fit)
			ro.save(path)

	print(f"\n[PROGRESS] fitting Readout on {args.device} done {now(True)}.\n")
	return


if __name__ == "__main__":
	warnings.filterwarnings("always")
	logging.captureWarnings(True)
	_main()
