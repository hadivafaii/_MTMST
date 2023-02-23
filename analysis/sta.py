from utils.generic import *


def compute_sta(
		lags: int,
		good: np.ndarray,
		stim: np.ndarray,
		spks: np.ndarray,
		verbose: bool = False, ):
	shape = stim.shape
	nc = spks.shape[-1]
	tgt = (nc, ) + (1, ) * len(shape)
	sta = np.zeros((nc, lags) + shape[1:])
	idxs = good.copy()
	idxs = idxs[idxs > lags]
	for t in tqdm(idxs, disable=not verbose):
		x = stim[t - lags: t]
		x = np.expand_dims(x, 0)
		x = np.repeat(x, nc, axis=0)
		y = spks[t].reshape(tgt)
		sta += x * y
	sta /= len(idxs)
	return sta
