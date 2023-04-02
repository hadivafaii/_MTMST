from figures.fighelper import *
from sklearn import linear_model
from sklearn.feature_selection import mutual_info_regression


def regress(
		z: np.ndarray,
		g: np.ndarray,
		z_tst: np.ndarray,
		g_tst: np.ndarray,
		parallel: bool = True,
		n_jobs: int = -1, ):
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
	# linear regression
	lr = linear_model.LinearRegression().fit(z, g)
	r = 1 - sp_dist.cdist(
		XA=g_tst.T,
		XB=lr.predict(z_tst).T,
		metric='correlation',
	)
	return mi, r, lr
