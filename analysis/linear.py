from .helper import *
from .opticflow import VelField


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
		self.fn = getattr(linear_model, category)
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
		self.kf = KFold(
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

	def fit(self, **kwargs):
		kwargs = setup_kwargs(self.defaults, kwargs)
		for a in self.alphas:
			kwargs['alpha'] = a
			model = self.fn(**kwargs).fit(flatten_stim(self.x), self.y)
			self.kers[a] = VelField(model.coef_.reshape(self.x.shape[1:]))
			self.models[a] = model
			if self.x_tst is not None:
				pred = model.predict(flatten_stim(self.x_tst))
				self.r_tst[a] = sp_stats.pearsonr(self.y_tst, pred)[0]
				self.r2_tst[a] = r2_score(self.y_tst, pred) * 100
				self.preds[a] = pred
		if self.df is None:
			self._fit_df(**kwargs)
		return self

	def _fit_df(self, **kwargs):
		df = []
		for a in self.alphas:
			kwargs['alpha'] = a
			_, nnll, r2, r = self._fit_folds(**kwargs)
			df.append({
				'alpha': [a] * self.kf.n_splits,
				'fold': range(self.kf.n_splits),
				'nnll': nnll,
				'r2': r2,
				'r': r,
			})
		df = pd.DataFrame(merge_dicts(df))
		df = df.groupby(['alpha']).mean()
		df = df.drop(columns=['fold'])
		self.df = df
		return

	def _fit_folds(self, **kwargs):
		kers, nnll, r, r2 = [], [], [], []
		for trn, vld in self.kf.split(self.x):
			model = self.fn(**kwargs).fit(
				X=flatten_stim(self.x[trn]),
				y=self.y[trn],
			)
			kers.append(VelField(
				model.coef_.reshape(self.x.shape[1:])))
			pred = model.predict(flatten_stim(self.x[vld]))
			nnll.append(null_adj_ll(self.y[vld], np.maximum(0, pred)))
			r.append(sp_stats.pearsonr(self.y[vld], pred)[0])
			if self.x_tst is not None:
				pred = model.predict(flatten_stim(self.x_tst))
				r2.append(r2_score(self.y_tst, pred) * 100)
			else:
				r2.append(None)
		return kers, nnll, r2, r

	def show_pred(self):
		if not self.r2_tst:
			return
		fig, ax = create_figure(1, 1, (7, 3.5))
		ax.plot(self.y_tst, lw=1.8, color='k', label='true')
		for i, (a, r2) in enumerate(self.r2_tst.items()):
			lbl = r"$R^2 = $" + f"{r2:0.1f}%  ("
			lbl += r"$\alpha = $" + f"{a:0.3g})"
			ax.plot(self.preds[a], color=f'C{i}', label=lbl)
		ax.legend(fontsize=9)
		leg = ax.get_legend()
		if leg is not None:
			leg.set_bbox_to_anchor((1.0, 1.025))
		ax.grid()
		plt.show()
		return fig, ax


def flatten_stim(x):
	return x.reshape(len(x), -1)
