from figures.fighelper import *
from sklearn import linear_model
from sklearn.feature_selection import mutual_info_regression


def regress(z, g, z_tst, g_tst):
	# mi regression
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
