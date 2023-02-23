from utils_model import *


def setup_repeat_data(
		group: h5py.Group,
		lags: int,
		stim: str = 'stimR', ):
	if 'repeats' not in group:
		return None, None
	group = group['repeats']
	psth = np.array(group['psth_raw_all'], dtype=float)
	badspks = np.array(group['fix_lost_all'], dtype=bool)
	tstart = np.array(group['tind_start_all'], dtype=int)
	assert (tstart == tstart[0]).all()
	tstart = tstart[0]
	nc, _, length = psth.shape

	# stim
	stim = np.array(group[stim], dtype=float)
	intvl = range(tstart[1], tstart[1] + length)
	src = time_embed(stim, lags, intvl)
	# spks
	spks = np.array(group['spksR'], dtype=float)
	tgt = np_nans(psth.shape)
	for i in range(nc):
		for trial, t in enumerate(tstart):
			s_ = range(t, t + length)
			tgt[i][trial] = spks[:, i][s_]
	tgt[badspks] = np.nan
	return src, tgt, intvl


def setup_supervised_data(
		lags: int,
		good: np.ndarray,
		stim: np.ndarray,
		spks: np.ndarray, ):
	assert len(stim) == len(spks), "must have same nt"
	idxs = good.copy()
	idxs = idxs[idxs > lags]
	src = time_embed(stim, lags, idxs)
	tgt = spks[idxs]
	assert len(src) == len(tgt), "must have same length"
	return src, tgt


def time_embed(x, lags, idxs=None):
	assert len(x) > lags
	if idxs is None:
		idxs = range(lags, len(x))
	x_emb = []
	for t in idxs:
		x_emb.append(np.expand_dims(
			x[t - lags: t], axis=0))
	return np.concatenate(x_emb)
