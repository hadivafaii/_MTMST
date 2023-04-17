from .helper import *
from base.dataset import load_ephys
from vae.train_vae import TrainerVAE
from .linear import compute_sta, LinearModel
from base.common import get_act_fn, nn, F, load_model_lite

_ATTRS = [
	'root', 'expt', 'n_pcs', 'n_lags', 'n_top_pix',
	'rescale', 'kws_hf', 'kws_xt', 'normalize', 'dtype',
]
_FIT = [
	'sta', 'temporal', 'spatial',
	'best_lags', 'best_pix_all', 'sorted_pix',
	'mu', 'sd', 'pca', 'glm', 'best_pix', 'perf', 'df',
]


class ReadoutGLM(object):
	def __init__(
			self,
			root: str,
			expt: str,
			tr: TrainerVAE = None,
			n_pcs: int = 500,
			n_lags: int = 13,
			n_top_pix: int = 8,
			rescale: float = 2.0,
			normalize: bool = True,
			dtype: str = 'float32',
			verbose: bool = False,
			**kwargs,
	):
		super(ReadoutGLM, self).__init__()
		self.root = root
		self.expt = expt
		self.tr = tr
		self.n_pcs = n_pcs
		self.n_lags = n_lags
		self.n_top_pix = n_top_pix
		self.rescale = rescale
		self.kws_hf = {
			k: kwargs[k] if k in kwargs else v for k, v
			in dict(dim=17, sres=1, radius=7).items()
		}
		self.kws_xt = {
			k: kwargs[k] if k in kwargs else v for k, v in
			dict(scale=4, pool='max', act_fn='swish').items()
		}
		self.normalize = normalize
		self.verbose = verbose
		self.dtype = dtype
		# load neuron attributes
		self.has_repeats, self.nc = None, None
		self.stim, self.stim_r = None, None
		self.spks, self.spks_r = None, None
		self.good, self.good_r = None, None
		# fitted attributes
		self.best_pix = {}
		self.pca, self.glm = {}, {}
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
			alphas: List[float] = None,
			**kwargs, ):
		kws_glm = dict(
			category='PoissonRegressor',
			alphas=alphas if alphas else
			np.logspace(-7, 1, num=9),
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
			kws_glm.update(data)
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				glm = LinearModel(**kws_glm).fit_linear(
					fit_df=not self.has_repeats, **kwargs)
				# TODO: for final results take out full=False
			i, j = pix
			if self.has_repeats:
				a, r = max(
					glm.df['r_tst'].items(),
					key=lambda t: t[1],
				)
				perf_r[i, j] = glm.df['r_tst'].max()
				perf_r2[i, j] = glm.df['r2_tst'].max()
			else:
				a, r = max(
					glm.df['r'].items(),
					key=lambda t: t[1],
				)
				perf_r[i, j] = r

			if r > best_r:
				best_r = r
				best_a = a
				self.perf[idx] = r
				self.df[idx] = glm.df
				self.glm[idx] = glm.models[a]
				self.best_pix[idx] = (i, j)
				self.pca[idx] = pc

			if self.verbose:
				msg = '-' * 80
				msg += f"\n{self.expt}, "
				msg += f"neuron # {idx}; "
				msg += f"pix: (i, j) = ({i}, {j})"
				print(msg)
				print(glm.df)
				glm.show_pred()
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
		)
		ftr, ftr_p = push(stim=x, **kws)
		if full:
			dims = (0, -2, -1)
			var = np.var(ftr, axis=dims)
			mu2 = np.mean(ftr, axis=dims) ** 2
			stats = dict(
				var=var,
				mu2=mu2,
				snr2=var / mu2,
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
		path = '/home/hadi/Documents/MTMST/results'
		path = pjoin(path, 'GLM', fit_name)
		# pickle
		file = f"{self.name()}.pkl"
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
		path = pjoin(path, 'GLM', fit_name)
		path_tr = pjoin(path, 'Trainer')
		os.makedirs(path_tr, exist_ok=True)

		# save trainer
		self.tr.save(path_tr)
		self.tr.cfg.save(path_tr)
		self.tr.model.cfg.save(path_tr)
		# save pickle
		fname = self.name()
		save_obj(
			obj=self.state_dict(),
			file_name=fname,
			save_dir=path,
			mode='pkl',
			verbose=self.verbose,
		)
		return

	def state_dict(self):
		return {k: getattr(self, k) for k in _ATTRS + _FIT}

	def name(self):
		return f"{self.root}-{self.expt}"

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
		)
		ftr, _ = push(stim=self.stim, **kws)
		ftr_r, _ = push(stim=self.stim_r, **kws)
		# normalize?
		self.mu = ftr.mean() if self.normalize else 1
		self.sd = ftr.std() if self.normalize else 0
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

	# TODO
	parser.add_argument(
		"tmp1",
		help='tmp 1',
		type=int,
	)
	parser.add_argument(
		"--tmp",
		help='tmp',
		type=int,
		default=1,
	)
	return parser.parse_args()


def _main():
	args = _setup_args()
	print(args)

	# TODO: create a saving framework
	#  Then run this

	# stuff
	print(f"\n[PROGRESS] fitting ReadoutGLM done {now(True)}.\n")
	return


if __name__ == "__main__":
	_main()
