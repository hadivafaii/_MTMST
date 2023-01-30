from .helper import *
from .opticflow import VelField
from sklearn.linear_model import PoissonRegressor


class GLM(Obj):
	def __init__(
			self,
			x: np.ndarray,
			y: np.ndarray,
			x_tst: np.ndarray = None,
			y_tst: np.ndarray = None,
			alphas: Iterable[float] = None,
			n_fols: int = 5,
			seed: int = 42,
			**kwargs,
	):
		super(GLM, self).__init__(**kwargs)
		self.x = x
		self.y = y
		self.x_tst = x_tst
		self.y_tst = y_tst
		if alphas is None:
			alphas = [10, 20, 50]
		assert isinstance(alphas, Iterable)
		self.alphas = alphas
		self.kf = KFold(
			n_splits=n_fols,
			random_state=seed,
			shuffle=True,
		)
		self.kers = {}
		self.pred = {}
		self.r2 = {}

	def fit(self, **kwargs):
		defaults = get_default_params(PoissonRegressor)
		kwargs = setup_kwargs(defaults, kwargs)
		for a in self.alphas:
			kwargs['alpha'] = a
			glm = PoissonRegressor(**kwargs)
			glm.fit(flatten_stim(self.x), self.y)
			self.kers[a] = VelField(glm.coef_.reshape(self.x.shape[1:]))
			if self.x_tst is not None:
				pred = glm.predict(flatten_stim(self.x_tst))
				self.r2[a] = r2_score(self.y_tst, pred) * 100
				self.pred[a] = pred
		return self

	def perf_df(self):
		results = []
		for a in self.alphas:
			_, nnll, r2 = self._fit_folds(alpha=a)
			results.append({
				'alpha': [a] * self.kf.n_splits,
				'fold': range(self.kf.n_splits),
				'nnll': nnll,
				'r2': r2,
			})
		return pd.DataFrame(merge_dicts(results))

	def _fit_folds(self, **kwargs):
		defaults = get_default_params(PoissonRegressor)
		kwargs = setup_kwargs(defaults, kwargs)
		kers, nnll, r2 = [], [], []
		for trn, vld in self.kf.split(self.x):
			glm = PoissonRegressor(**kwargs)
			glm.fit(
				X=flatten_stim(self.x[trn]),
				y=self.y[trn],
			)
			kers.append(VelField(
				glm.coef_.reshape(self.x.shape[1:])))
			pred = glm.predict(flatten_stim(self.x[vld]))
			nnll.append(null_adj_ll(self.y[vld], pred))
			if self.x_tst is not None:
				pred = glm.predict(flatten_stim(self.x_tst))
				r2.append(r2_score(self.y_tst, pred) * 100)
			else:
				r2.append(None)
		return kers, nnll, r2

	def show_pred(self):
		if not self.r2:
			return
		fig, ax = create_figure(1, 1, (7, 3.5))
		ax.plot(self.y_tst, lw=1.8, label='true')
		for i, (a, r2) in enumerate(self.r2.items()):
			lbl = r"$R^2 = $" + f"{r2:0.1f}%  ("
			lbl += r"$\alpha = $" + f"{a:0.1f})"
			ax.plot(self.pred[a], label=lbl)
		ax.legend(fontsize=10)
		ax.grid()
		plt.show()
		return fig, ax


def flatten_stim(x):
	return x.reshape(len(x), -1)
