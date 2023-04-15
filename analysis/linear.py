from .helper import *
from .opticflow import VelField
from sklearn.feature_selection import mutual_info_regression


def mi_analysis(
		z: np.ndarray,
		g: np.ndarray,
		n_bins: int = 20,
		parallel: bool = True,
		n_jobs: int = -1,):
	# mi regression
	if parallel:
		with joblib.parallel_backend('multiprocessing'):
			mi = joblib.Parallel(n_jobs=n_jobs)(
				joblib.delayed(mutual_info_regression)
				(g, z[:, i]) for i in range(z.shape[-1])
			)
		mi = np.stack(mi).T
	else:
		mi = np.zeros((g.shape[-1], z.shape[-1]))
		for i in range(len(mi)):
			mi[i] = mutual_info_regression(z, g[:, i])
	# mi normalized (discrete)
	mi_normalized = discrete_mutual_info(
		z=z,
		g=g,
		axis=1,
		n_bins=n_bins,
		parallel=parallel,
		n_jobs=n_jobs,
	)
	output = {
		'mi': mi,
		'mi_norm': mi_normalized,
		'mig': compute_mig(mi_normalized),
	}
	return output


def regress(
		z: np.ndarray,
		g: np.ndarray,
		z_tst: np.ndarray,
		g_tst: np.ndarray,
		process: bool = True, ):
	if process:
		mu, sd = z.mean(), z.std()
		z = (z - mu) / sd
		z_tst = (z_tst - mu) / sd
	# linear regression
	lr = sk_linear.LinearRegression().fit(z, g)
	g_pred = lr.predict(z_tst)
	# DCI
	w = np.abs(lr.coef_)
	w *= z.std(0).reshape(1, -1)
	w /= g.std(0).reshape(-1, 1)
	d, c = compute_dci(w)
	output = {
		'r2': sk_metric.r2_score(
			y_true=g_tst,
			y_pred=g_pred,
			multioutput='raw_values'),
		'r': 1 - sp_dist.cdist(
			XA=g_tst.T,
			XB=g_pred.T,
			metric='correlation'),
		'd': d,
		'c': c,
	}
	return output


def compute_mig(mi_normalized: np.ndarray, axis: int = 0):
	assert mi_normalized.ndim == 2
	n_factors = mi_normalized.shape[axis]
	mig = np.zeros(n_factors)
	for i in range(n_factors):
		a = mi_normalized.take(i, axis)
		inds = np.argsort(a)[::-1]
		mig[i] = a[inds[0]] - a[inds[1]]
	return mig


def compute_dci(w: np.array):
	# p_disentang
	denum = w.sum(0, keepdims=True)
	denum[denum == 0] = np.nan
	p_disentang = w / denum
	# p_complete
	denum = w.sum(1, keepdims=True)
	denum[denum == 0] = np.nan
	p_complete = w / denum
	# compute D and C
	d_i = 1 - entropy_normalized(p_disentang, 0)
	c_mu = 1 - entropy_normalized(p_complete, 1)
	rho = w.sum(0) / w.sum()
	d = np.nansum(d_i * rho)
	c = np.nanmean(c_mu)
	return d, c


class LinearModel(Obj):
	def __init__(
			self,
			category: str,
			x: np.ndarray,
			y: np.ndarray,
			x_tst: np.ndarray = None,
			y_tst: np.ndarray = None,
			alphas: Iterable[float] = None,
			n_folds: int = 5,
			seed: int = 0,
			**kwargs,
	):
		super(LinearModel, self).__init__(**kwargs)
		self.fn = getattr(sk_linear, category)
		self.defaults = get_default_params(self.fn)
		if 'random_state' in self.defaults:
			self.defaults['random_state'] = seed
		self.category = category
		self.x = x
		self.y = y
		self.x_tst = x_tst
		self.y_tst = y_tst
		if alphas is None:
			alphas = [0.1, 1, 10, 100]
		assert isinstance(alphas, Iterable)
		self.alphas = alphas
		self.kf = sk_modselect.KFold(
			n_splits=n_folds,
			random_state=seed,
			shuffle=True,
		)
		self.models = {}
		self.kers = {}
		self.preds = {}
		self.r_tst = {}
		self.r2_tst = {}
		self.df = None

		if self.verbose:
			msg = f"Category: '{self.category}', "
			msg += f"default params:\n{self.defaults}"
			print(msg)

	def fit_linear(self, fit_df: bool = True, **kwargs):
		kwargs = setup_kwargs(self.defaults, kwargs)
		for a in self.alphas:
			kwargs['alpha'] = a
			model = self.fn(**filter_kwargs(self.fn, kwargs))
			model.fit(flatten_stim(self.x), self.y)
			kernel = model.coef_.reshape(self.x.shape[1:])
			try:
				self.kers[a] = VelField(kernel)
			except AssertionError:
				self.kers[a] = kernel
			self.models[a] = model
			if self.x_tst is not None:
				pred = model.predict(flatten_stim(self.x_tst))
				self.r_tst[a] = sp_stats.pearsonr(self.y_tst, pred)[0]
				self.r2_tst[a] = sk_metric.r2_score(self.y_tst, pred) * 100
				self.preds[a] = pred
		if self.df is None and fit_df:
			self.fit_df(**kwargs)
		return self

	def fit_df(self, **kwargs):
		df = []
		for a in self.alphas:
			kwargs['alpha'] = a
			nnll, r2, r = self._fit_folds(**kwargs)
			df.append({
				'alpha': [a] * len(r),
				'fold': r.keys(),
				'r': r.values(),
				'r2': r2.values(),
				'nnll': nnll.values(),
			})
		df = pd.DataFrame(merge_dicts(df))
		df = df.groupby(['alpha']).mean()
		df = df.drop(columns=['fold'])
		self.df = df
		return self

	def _fit_folds(self, full: bool = True, **kwargs):
		nnll, r, r2 = {}, {}, {}
		for fold, (trn, vld) in enumerate(self.kf.split(self.x)):
			if not full and fold > 0:
				continue
			model = self.fn(**filter_kwargs(self.fn, kwargs))
			model.fit(flatten_stim(self.x[trn]), self.y[trn])
			pred = model.predict(flatten_stim(self.x[vld]))
			nnll[fold] = null_adj_ll(self.y[vld], np.maximum(0, pred))
			r[fold] = sp_stats.pearsonr(self.y[vld], pred)[0]
			if self.x_tst is not None:
				pred = model.predict(flatten_stim(self.x_tst))
				r2[fold] = sk_metric.r2_score(self.y_tst, pred) * 100
			else:
				r2[fold] = None
		return nnll, r2, r

	def show_pred(self, figsize=(7, 3.5)):
		if not self.r2_tst:
			return
		fig, ax = create_figure(1, 1, figsize)
		ax.plot(self.y_tst, lw=1.8, color='k', label='true')
		for i, (a, r2) in enumerate(self.r2_tst.items()):
			lbl = r"$R^2 = $" + f"{r2:0.1f}%  ("
			lbl += r"$\alpha = $" + f"{a:0.2g})"
			ax.plot(self.preds[a], color=f'C{i}', label=lbl)
		ax.legend(fontsize=9)
		leg = ax.get_legend()
		if leg is not None:
			leg.set_bbox_to_anchor((1.0, 1.025))
		ax.grid()
		plt.show()
		return fig, ax


def compute_sta_remove(
		n_lags: int,
		good: np.ndarray,
		stim: np.ndarray,
		spks: np.ndarray,
		zscore: bool = True,
		verbose: bool = False, ):
	assert n_lags >= 0
	nc = spks.shape[-1]
	shape = stim.shape[1:]
	stim = flatten_stim(stim)
	if zscore:
		stim = sp_stats.zscore(stim)
	stim = np.expand_dims(stim, 0)
	inds = good.copy()
	inds = inds[inds > n_lags]
	# compute sta
	sta = np.zeros((nc, n_lags + 1, np.prod(shape)))
	for t in tqdm(inds, disable=not verbose):
		# zero n_lags allowed:
		x = stim[:, t - n_lags: t + 1]
		y = spks[t].reshape((-1, 1, 1))
		sta += x * y
	# divide by # spks
	n = spks[inds].sum(0)
	n = n.reshape((-1, 1, 1))
	sta /= n
	# reshape back to original
	shape = (nc, n_lags + 1, *shape)
	sta = sta.reshape(shape)
	return sta


def compute_sta(
		n_lags: int,
		good: np.ndarray,
		stim: np.ndarray,
		spks: np.ndarray,
		zscore: bool = True,
		verbose: bool = False, ):
	assert n_lags >= 0
	shape = stim.shape
	nc = spks.shape[-1]
	sta = np.zeros((nc, n_lags+1, *shape[1:]))
	shape = (nc,) + (1,) * len(shape)
	if zscore:
		stim = sp_stats.zscore(stim)
	inds = good.copy()
	inds = inds[inds > n_lags]
	for t in tqdm(inds, disable=not verbose):
		# zero n_lags allowed:
		x = stim[t - n_lags: t + 1]
		x = np.expand_dims(x, 0)
		y = spks[t].reshape(shape)
		sta += x * y
	n = spks[inds].sum(0)
	n = n.reshape(shape)
	sta /= n
	return sta
