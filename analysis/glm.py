from .helper import *
from base.dataset import load_ephys
from vae.train_vae import TrainerVAE
from .linear import compute_sta, LinearModel
from base.common import get_act_fn, nn, F, load_model_lite, load_model

_ATTRS = [
	'root', 'expt', 'glm', 'n_pcs', 'n_lags', 'n_top_pix',
	'rescale', 'kws_hf', 'kws_xt', 'normalize', 'dtype',
]
_FIT = [
	'sta', 'temporal', 'spatial',
	'best_lags', 'best_pix_all', 'sorted_pix',
	'mu', 'sd', 'pca', 'mod', 'best_pix', 'perf', 'df',
]


class ReadoutGLM(object):
	def __init__(
			self,
			root: str,
			expt: str,
			tr: TrainerVAE,
			n_pcs: int = 500,
			n_lags: int = 20,
			n_top_pix: int = 8,
			rescale: float = 2.0,
			dtype: str = 'float32',
			normalize: bool = True,
			verbose: bool = False,
			**kwargs,
	):
		super(ReadoutGLM, self).__init__()
		self.tr = tr
		self.root = root
		self.expt = expt
		self.n_pcs = n_pcs
		self.n_lags = n_lags
		self.n_top_pix = n_top_pix
		self.rescale = rescale
		self.kws_hf = {
			k: kwargs[k] if k in kwargs else v for k, v
			in dict(dim=17, sres=1, radius=8.0).items()
		}
		self.kws_push = {
			k: kwargs[k] if k in kwargs else v for k, v
			in dict(which='enc', use_ema=False).items()
		}
		self.kws_xt = {
			k: kwargs[k] if k in kwargs else v for k, v in
			dict(scale=4, pool='max', act_fn='swish').items()
		}
		self.normalize = normalize
		self.verbose = verbose
		self.dtype = dtype
		self.glm = None
		# neuron attributes
		self.has_repeats, self.nc = None, None
		self.stim, self.stim_r = None, None
		self.spks, self.spks_r = None, None
		self.good, self.good_r = None, None
		# fitted attributes
		self.best_pix = {}
		self.pca, self.mod = {}, {}
		self.perf, self.df = {}, {}

	def fit_readout(self):
		self.load_neuron()
		self._xtract()
		self._sta()
		self._best_lags()
		self._best_pix()
		return self

	def fit_neuron(
			self,
			idx: int,
			glm: bool = False,
			alphas: List[float] = None,
			**kwargs, ):
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
				np.logspace(-1, 7, num=9),
			)
		best_a = None
		best_r = -np.inf
		perf_r = np.zeros(self.spatial.shape[1:])
		perf_r2 = np.zeros(self.spatial.shape[1:])
		for pix in self.sorted_pix[idx]:
			pc = sk_decomp.PCA(
				n_components=self.n_pcs,
				svd_solver='full',
			)
			data = self.get_data(idx, pix=pix)
			data['x'] = pc.fit_transform(data['x'])
			if self.has_repeats:
				data['x_tst'] = pc.transform(data['x_tst'])
			kws_model.update(data)
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				linmod = LinearModel(**kws_model).fit_linear(
					xv_folds=not self.has_repeats, **kwargs)
				# TODO: for final results take out full=False
			i, j = pix
			if self.has_repeats:
				a, r = max(
					linmod.df['r_tst'].items(),
					key=lambda t: t[1],
				)
				perf_r[i, j] = linmod.df['r_tst'].max()
				perf_r2[i, j] = linmod.df['r2_tst'].max()
			else:
				a, r = max(
					linmod.df['r'].items(),
					key=lambda t: t[1],
				)
				perf_r[i, j] = r

			if r > best_r:
				best_r = r
				best_a = a
				self.perf[idx] = r
				self.df[idx] = linmod.df
				self.mod[idx] = linmod.models[a]
				self.best_pix[idx] = (i, j)
				self.pca[idx] = pc

			if self.verbose:
				msg = '-' * 80
				msg += f"\n{self.expt}, "
				msg += f"neuron # {idx}; "
				msg += f"pix: (i, j) = ({i}, {j})"
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
			pix: Tuple[int, int] = None, ):
		if ftr is None:
			ftr = self.ftr
		if ftr_r is None:
			ftr_r = self.ftr_r
		if pix is None:
			pix = self.best_pix[idx]
		i, j = pix
		kws = dict(
			lag=self.best_lags[idx],
			x=ftr[..., i, j],
			y=self.spks[:, idx],
			good=self.good,
		)
		if self.has_repeats:
			kws.update(dict(
				x_tst=ftr_r[..., i, j],
				y_tst=self.spks_r[idx],
				good_tst=self.good_r,
			))
		return setup_data(**kws)

	def load_neuron(self):
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
		self.nc = spks.shape[1]
		self.stim, self.stim_r = stim, stim_r
		self.spks, self.spks_r = spks, spks_r
		self.good, self.good_r = np.where(mask)[0], good_r
		f.close()
		if self.verbose:
			print('[PROGRESS] neural data loaded')
		return self

	def load(self, fit_name: str, device: str):
		path = 'Documents/MTMST/results'
		path = pjoin(os.environ['HOME'], path)
		root_name, pickle_name = self.name()
		path = pjoin(path, root_name, fit_name)
		# pickle
		file = f"{pickle_name}.pkl"
		file = pjoin(path, file)
		with (open(file, 'rb')) as f:
			pkl = pickle.load(f)
		for k, v in pkl.items():
			setattr(self, k, v)
		# Trainer
		if self.tr is None:
			path = pjoin(path, 'Trainer')
			self.tr, _ = load_model_lite(
				path=path,
				device=device,
				verbose=self.verbose,
			)
		return self

	def save(self, fit_name: str):
		path = self.tr.model.cfg.results_dir
		root_name, pickle_name = self.name()
		path = pjoin(path, root_name, fit_name)
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
			file_name=pickle_name,
			save_dir=path,
			mode='pkl',
			verbose=self.verbose,
		)
		return

	def state_dict(self):
		return {k: getattr(self, k) for k in _ATTRS + _FIT}

	def name(self):
		root_name = 'GLM' if self.glm else 'Ridge'
		pickle_name = f"{self.root}-{self.expt}"
		return root_name, pickle_name

	def show(self, idx: int):
		fig, axes = create_figure(
			1, 2, (9, 2.7),
			width_ratios=[3, 1],
			constrained_layout=True,
		)
		axes[0].plot(self.temporal[idx] * 100, marker='o')
		axes[0].axvline(
			self.n_lags - self.best_lags[idx],
			color='r',
			ls='--',
			label=f'best lag = {self.best_lags[idx]}',
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
		axes[0].tick_params(axis='x', rotation=-90, labelsize=9)
		axes[0].legend(fontsize=11)
		axes[0].grid()

		sns.heatmap(
			data=self.spatial[idx] * 1000,
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

	def _sta(self):
		self.sta = compute_sta(
			stim=self.ftr,
			good=self.good,
			spks=self.spks,
			n_lags=self.n_lags,
			verbose=self.verbose,
			zscore=True,
		)
		if self.verbose:
			print('[PROGRESS] sta computed')
		return

	def _best_lags(self):
		self.temporal = np.mean(self.sta ** 2, axis=(2, 3, 4))
		self.best_lags = np.argmax(self.temporal[:, ::-1], axis=1)
		if self.verbose:
			print('[PROGRESS] best lag estimated')
		return

	def _best_pix(self):
		# best pix per lag
		shape = (self.nc, self.n_lags + 1, 2)
		self.best_pix_all = np.zeros(shape, dtype=int)
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
			self.best_pix_all[idx, t] = i, j
		# best pix
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
	m = tr.select_model(use_ema)
	nf_enc, nf_dec = m.ftr_sizes()
	nf_enc = sum(nf_enc.values())
	nf_dec = sum(nf_dec.values())
	if which == 'enc':
		nf = nf_enc
	elif which == 'dec':
		nf = nf_dec
	elif which == 'both':
		nf = nf_enc + nf_dec
	else:
		raise NotImplementedError(which)

	s = kws_process['scale']
	shape = (len(stim), nf, s, s)
	x = np.empty(shape, dtype=dtype)
	if max_pool:
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
		ftr = tr.to(stim[a:b])
		ftr = m.xtract_ftr(ftr, full=True)[-1]
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


def _setup_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"name",
		help='fit name',
		type=str,
	)
	parser.add_argument(
		"device",
		help='cuda:n',
		type=str,
	)
	parser.add_argument(
		"--model_name",
		help='which VAE to load',
		default=None,
		type=str,
	)
	parser.add_argument(
		"--fit_name",
		help='which VAE fit to load',
		default=None,
		type=str,
	)
	parser.add_argument(
		"--checkpoint",
		help='checkpoint',
		default=-1,
		type=int,
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
		default=20,
		type=int,
	)
	parser.add_argument(
		"--n_top_pix",
		help='# top pixels to loop over',
		default=8,
		type=int,
	)
	parser.add_argument(
		"--rescale",
		help='HyperFlow stim rescale',
		default=2.0,
		type=float,
	)
	parser.add_argument(
		"--radius",
		help='HyperFlow stim radius',
		default=8.0,
		type=float,
	)
	parser.add_argument(
		"--normalize",
		help='normalize before PCA?',
		default=True,
		type=bool,
	)
	parser.add_argument(
		"--which",
		help="which to use: {'enc', 'dec'}",
		default='enc',
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
	# VAE related
	parser.add_argument(
		"--n_ch",
		help='# channels',
		default=32,
		type=int,
	)
	parser.add_argument(
		"--n_enc_cells",
		help='# enc cells',
		default=2,
		type=int,
	)
	parser.add_argument(
		"--n_enc_nodes",
		help='# enc nodes',
		default=2,
		type=int,
	)
	parser.add_argument(
		"--n_dec_cells",
		help='# dec cells',
		default=2,
		type=int,
	)
	parser.add_argument(
		"--n_dec_nodes",
		help='# dec nodes',
		default=1,
		type=int,
	)
	parser.add_argument(
		"--n_pre_cells",
		help='# preprocessing cells',
		default=3,
		type=int,
	)
	parser.add_argument(
		"--n_pre_blocks",
		help='# preprocessing blocks',
		default=1,
		type=int,
	)
	parser.add_argument(
		"--n_post_cells",
		help='# postprocessing cells',
		default=3,
		type=int,
	)
	parser.add_argument(
		"--n_post_blocks",
		help='# postprocessing blocks',
		default=1,
		type=int,
	)
	parser.add_argument(
		"--n_latent_scales",
		help='# latent scales',
		default=3,
		type=int,
	)
	parser.add_argument(
		"--n_latent_per_group",
		help='# latents per group',
		default=13,
		type=int,
	)
	parser.add_argument(
		"--n_groups_per_scale",
		help='# groups per scale',
		default=12,
		type=int,
	)
	parser.add_argument(
		"--ada_groups",
		help='adaptive latent groups',
		default=True,
		type=bool,
	)
	parser.add_argument(
		"--compress",
		help='compress latent space',
		default=True,
		type=bool,
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
	print(args)

	# TODO: if model name is None, then create a Reservoir
	#  else: replace the default VAE config in args using the loaded VAE
	tr, metadata = load_model(
		model_name='fixate1_k-32_z-13x[3,6,12]_enc(1x3)-dec(1x2)-pre(1x3)-post(1x3)',
		fit_name='ep200-b500-lr(0.002)_beta(0.1:0x0.3)_lamb(0.001)_gr(400.0)_(2023_04_18,16:48)',
		device=args.device,
		checkpoint=70,
	)
	nf = sum(tr.model.ftr_sizes()[0].values())
	fit_name = '_'.join([
		args.name,
		f"nf-{nf}",
		f"({now(True)})",
	])
	print(f"\nname: {fit_name}")

	kws = dict(
		tr=tr,
		root='YUWEI',
		n_pcs=args.n_pcs,
		n_lags=args.n_lags,
		n_top_pix=args.n_top_pix,
		rescale=args.rescale,
		normalize=args.normalize,
		verbose=args.verbose,
		# kwargs
		which=args.which,
		radius=args.radius,
		use_ema=args.use_ema,
	)
	for expt, useful in tr.model.cfg.useful_yuwei.items():
		ro = ReadoutGLM(expt=expt, **kws).fit_readout()
		for idx in useful:
			perf_r, _ = ro.fit_neuron(idx, args.glm, full=True)
		ro.save(fit_name)
	# if args.verbose:
	# ro.show(0)
	# sns.heatmap(perf_r, square=True, annot=True)
	# plt.show()
	# TODO: save args as JSON file in the directory .../Ridge/name_date/

	print(f"\n[PROGRESS] fitting ReadoutGLM done {now(True)}.\n")
	return


if __name__ == "__main__":
	_main()
