from .utils_model import *
from analysis.opticflow import HyperFlow
from torch.utils.data.dataset import Dataset


# noinspection PyUnresolvedReferences
class ROFL(Dataset):
	def __init__(
			self,
			path: str,
			mode: str,
			device: torch.device = None,
	):
		# attributes
		self.attrs = np.load(
			pjoin(path, 'attrs.npy'),
			allow_pickle=True,
		).item()
		self.f = self.attrs.pop('f')
		self.f_aux = self.attrs.pop('f_aux')
		# mode = trn/vld/tst
		path = pjoin(path, mode)
		kws = dict(mmap_mode='r')
		# generative factors
		self.g = np.load(pjoin(path, 'g.npy'), **kws)
		self.g_aux = np.load(pjoin(path, 'g_aux.npy'), **kws)
		# data & norm
		self.x = np.load(pjoin(path, 'x.npy'), **kws)
		self.norm = np.load(pjoin(path, 'norm.npy'), **kws)
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

	def __len__(self):
		return len(self.x)

	def __getitem__(self, i):
		return self.x[i], self.norm[i]


def setup_repeat_data(
		group: h5py.Group,
		kws_hf: dict = None, ):
	if not group.attrs.get('has_repeats'):
		return None, None, None
	kws_hf = kws_hf if kws_hf else {
		'dim': 32,
		'sres': 1,
		'radius': 8,
	}
	psth = np.array(group['psth_raw_all'], dtype=float)
	badspks = np.array(group['fix_lost_all'], dtype=bool)
	tstart = np.array(group['tind_start_all'], dtype=int)
	assert (tstart == tstart[0]).all()
	tstart = tstart[0]
	nc, _, length = psth.shape
	intvl = range(tstart[1], tstart[1] + length)

	# stim
	hf = HyperFlow(
		params=np.array(group['hf_paramsR']),
		center=np.array(group['hf_centerR']),
		**kws_hf,
	)
	stim = hf.compute_hyperflow()
	stim = stim[range(intvl.stop)]
	intvl = np.array(intvl)

	# spks
	_spks = np.array(group['spksR'], dtype=float)
	spks = np_nans(psth.shape)
	for i in range(nc):
		for trial, t in enumerate(tstart):
			s_ = range(t, t + length)
			spks[i][trial] = _spks[:, i][s_]
	spks[badspks] = np.nan

	return stim, spks, intvl


def load_ephys(
		group: h5py.Group,
		kws_hf: dict = None,
		rescale: float = 2.0,
		dtype: str = 'float32', ):
	kws_hf = kws_hf if kws_hf else {
		'dim': 17,
		'sres': 1,
		'radius': 8.0,
	}
	hf = HyperFlow(
		params=np.array(group['hf_params']),
		center=np.array(group['hf_center']),
		**kws_hf,
	)
	stim = hf.compute_hyperflow(dtype=dtype)
	spks = np.array(group['spks'], dtype=float)
	if 'badspks' in group:
		mask = ~np.array(group['badspks'], dtype=bool)
	else:
		mask = np.ones(len(spks), dtype=bool)
	stim_r, spks_r, good_r = setup_repeat_data(
		group=group, kws_hf=kws_hf)

	if rescale is not None:
		stim_scale = np.max(np.abs(stim))
		stim *= rescale / stim_scale
		if stim_r is not None:
			stim_r *= rescale / stim_scale

	return stim, spks, mask, stim_r, spks_r, good_r


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


def simulation_combos():
	combos = [('fixate', i) for i in [0, 1, 2, 4]]
	combos += [('terrain', i) for i in [1, 2, 4, 8]]
	combos += [('transl', i) for i in [0, 2, 4]]
	combos += [('obj', i) for i in [1, 2, 4]]
	return combos


def _setup_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"n_tot",
		help='# frames total',
		type=int,
	)
	parser.add_argument(
		"--n_batch",
		help='# frames per batch',
		default=int(5e4),
		type=int,
	)
	parser.add_argument(
		"--dim",
		help='dimensionality',
		default=17,
		type=int,
	)
	parser.add_argument(
		"--min_obj_size",
		help='minimum acceptable object size',
		default=3.5,
		type=float,
	)
	parser.add_argument(
		"--dtype",
		help='dtype for alpha_dot',
		default='float32',
		type=str,
	)
	return parser.parse_args()


def _main():
	args = _setup_args()
	print(args)

	kws = dict(
		n=args.n_batch,
		dim=args.dim,
		fov=45.0,
		obj_r=0.25,
		obj_bound=0.97,
		obj_zlim=(0.5, 1.0),
		vlim_obj=(0.01, 1.0),
		vlim_slf=(0.01, 1.0),
		residual=False,
		z_bg=1.0,
		seed=0,
	)
	accept_n = {
		0: None,
		1: None,
		2: 1,
		4: 3,
		8: 5,
	}
	from utils.process import generate_simulation, save_simulation
	save_dir = '/home/hadi/Documents/MTMST/data'
	pbar = tqdm(simulation_combos())
	for category, n_obj in pbar:
		pbar.set_description(f"creating {category}{n_obj}")
		alpha_dot, g, g_aux, attrs = generate_simulation(
			total=args.n_tot,
			category=category,
			n_obj=n_obj,
			kwargs=kws,
			accept_n=accept_n,
			min_obj_size=args.min_obj_size,
			dtype=args.dtype,
		)
		save_simulation(
			save_dir=save_dir,
			x=alpha_dot,
			g=g,
			g_aux=g_aux,
			attrs=attrs,
		)
	print(f"\n[PROGRESS] saving datasets done ({now(True)}).\n")
	return


if __name__ == "__main__":
	_main()
