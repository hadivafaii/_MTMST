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
	# performance
	r = 1 - sp_dist.cdist(
		XA=g_tst.T,
		XB=g_pred.T,
		metric='correlation',
	)
	r2 = sk_metric.r2_score(
		y_true=g_tst,
		y_pred=g_pred,
		multioutput='raw_values',
	)
	r2[r2 <= 0] = np.nan
	# DCI
	w = np.abs(lr.coef_)
	w *= z.std(0).reshape(1, -1)
	w /= g.std(0).reshape(-1, 1)
	d, c = compute_dci(w)
	output = {
		'r': r,
		'r2': r2,
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
		assert isinstance(alphas, Collection)
		self.alphas = alphas
		self.kf = sk_modselect.KFold(
			n_splits=n_folds,
			random_state=seed,
			shuffle=True,
		)
		self.models = {}
		self.kers = {}
		self.preds = {}
		self._init_df()

		if self.verbose:
			msg = f"Category: '{self.category}', "
			msg += f"default params:\n{self.defaults}"
			print(msg)

	def fit_linear(self, xv_folds: bool = True, **kwargs):
		kwargs = setup_kwargs(self.defaults, kwargs)
		if xv_folds:
			self.fit_xv(**kwargs)
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
				r2 = sk_metric.r2_score(self.y_tst, pred) * 100
				r = sp_stats.pearsonr(self.y_tst, pred)[0]
				self.df.loc[a, 'r2_tst'] = r2
				self.df.loc[a, 'r_tst'] = r
				self.preds[a] = pred
		return self

	def fit_xv(self, **kwargs):
		for a in self.alphas:
			kwargs['alpha'] = a
			nnll, r = self._fit_folds(**kwargs)
			self.df.loc[a, 'nnll'] = np.nanmean(nnll)
			self.df.loc[a, 'r'] = np.nanmean(r)
		return self

	def _fit_folds(self, full: bool = True, **kwargs):
		nnll, r = [], []
		for fold, (trn, vld) in enumerate(self.kf.split(self.x)):
			if not full and fold > 0:
				continue
			model = self.fn(**filter_kwargs(self.fn, kwargs))
			model.fit(flatten_stim(self.x[trn]), self.y[trn])
			pred = model.predict(flatten_stim(self.x[vld]))
			nnll.append(null_adj_ll(self.y[vld], np.maximum(0, pred)))
			r.append(sp_stats.pearsonr(self.y[vld], pred)[0])
		return nnll, r

	def _init_df(self):
		fill_vals = [np.nan] * len(self.alphas)
		df = {
			'alpha': self.alphas,
			'r': fill_vals,
			'nnll': fill_vals,
		}
		if self.x_tst is not None:
			df.update({
				'r_tst': fill_vals,
				'r2_tst': fill_vals,
			})
		self.df = pd.DataFrame(df).set_index('alpha')
		return

	def show_pred(self, figsize=(9.0, 4.7)):
		if not self.preds:
			return
		fig, ax = create_figure(1, 1, figsize)
		ax.plot(self.y_tst, lw=1.8, color='k', label='true')
		for i, (a, pred) in enumerate(self.preds.items()):
			r2 = self.df.loc[a, 'r2_tst']
			lbl = r"$R^2 = $" + f"{r2:0.1f}%  ("
			lbl += r"$\alpha = $" + f"{a:0.2g})"
			ax.plot(pred, color=f'C{i}', label=lbl)
		ax.legend(fontsize=12)
		leg = ax.get_legend()
		if leg is not None:
			leg.set_bbox_to_anchor((1.0, 1.025))
		ax.grid()
		plt.show()
		return fig, ax


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
