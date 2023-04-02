from .utils_model import *
from analysis.opticflow import HyperFlow
from torch.utils.data.dataset import Dataset


class ROFL(Dataset):
	def __init__(
			self,
			g: h5py.Group,
			device: torch.device = None,
			transform=None,
	):
		self._init_factors(g)
		x = np.array(g['x'], dtype=float)
		self.x = np.transpose(
			x, (0, 3, 1, 2))
		self.norm = sp_lin.norm(
			x, axis=-1).sum(-1).sum(-1)
		if device is not None:
			self.x = torch.tensor(
				data=self.x,
				device=device,
				dtype=torch.float,
			)
			self.norm = torch.tensor(
				data=self.norm,
				device=device,
				dtype=torch.float,
			)
		self.transform = transform

	def __len__(self):
		return len(self.x)

	def __getitem__(self, i):
		x = self.x[i]
		n = self.norm[i]
		if self.transform is not None:
			x = self.transform(x)
		return x, n

	def _init_factors(self, g):
		fix = np.array(g['fix'], dtype=float)
		vel_slf = np.array(g['vel_slf'], dtype=float)
		vel_obj = np.array(g['vel_obj'], dtype=float)
		pos_obj = np.array(g['pos_obj'], dtype=float)
		pos_obj[0] -= fix[:, 0]
		pos_obj[1] -= fix[:, 1]
		factors = np_nans((len(fix), 11))
		factors[:, :2] = fix
		factors[:, 2:5] = vel_slf.T
		factors[:, 5:8] = vel_obj.T
		factors[:, 8:11] = pos_obj.T
		assert not np.isnan(factors).sum()
		self.factors = factors
		self.factor_names = {
			0: 'fix_x',
			1: 'fix_y',
			2: 'v_self_x',
			3: 'v_self_y',
			4: 'v_self_z',
			5: 'v_obj_x',
			6: 'v_obj_y',
			7: 'v_obj_z',
			8: 'pos_obj_x',
			9: 'pos_obj_y',
			10: 'pos_obj_z',
		}
		return


def setup_repeat_data(
		group: h5py.Group,
		lags: int = 24,
		hf_kws: dict = None,
		use_hf: bool = True, ):
	if 'repeats' not in group:
		return None, None, None
	hf_kws = hf_kws if hf_kws else {
		'size': 32,
		'sres': 1,
		'radius': 6,
	}
	group = group['repeats']
	psth = np.array(group['psth_raw_all'], dtype=float)
	badspks = np.array(group['fix_lost_all'], dtype=bool)
	tstart = np.array(group['tind_start_all'], dtype=int)
	assert (tstart == tstart[0]).all()
	tstart = tstart[0]
	nc, _, length = psth.shape

	# stim
	if use_hf:
		hf = HyperFlow(
			params=np.array(group['hyperflowR'])[:, 2:],
			center=np.array(group['hyperflowR'])[:, :2],
			**hf_kws,
		)
		stim = hf.compute_hyperflow()
	else:
		stim = np.array(group['stimR'], dtype=float)
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
