from .utils_model import *
from torch.utils.data.dataset import Dataset
from analysis.opticflow import ROFL, HyperFlow


# noinspection PyUnresolvedReferences
class ROFLDS(Dataset):
	def __init__(
			self,
			path: str,
			mode: str,
			device: torch.device = None,
	):
		# category & n_obj
		sim = path.split('/')[-1].split('_')[0]
		self.category = sim[:-1]
		self.n_obj = int(sim[-1])
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
		if self.category == 'obj':
			self.transform = _shift_mu
		else:
			self.transform = None

	def __len__(self):
		return len(self.x)

	def __getitem__(self, i):
		if self.transform is not None:
			x = self.transform(self.x[i])
		else:
			x = self.x[i]
		return x, self.norm[i]


def _shift_mu(x):
	return x - torch.mean(x)


def generate_simulation(
		category: str,
		n_obj: int,
		total: int,
		kwargs: dict,
		accept_n: dict,
		min_obj_size: int,
		dtype='float32', ):
	kws = kwargs.copy()
	kws['category'] = category
	kws['n_obj'] = n_obj
	kws['seed'] = 0

	shape = (total, kws['dim'], kws['dim'], 2)
	alpha_dot = np.empty(shape, dtype=dtype)
	g_all, g_aux_all = [], []

	cnt = 0
	while True:
		# generate
		of = ROFL(**kws).compute_coords()
		_ = of.compute_flow()
		# accept
		min_n_obj = accept_n[n_obj]
		accept = of.filter(
			min_obj_size=min_obj_size,
			min_n_obj=min_n_obj,
		)
		f, g, f_aux, g_aux = of.groundtruth_factors()
		ind = range(cnt, min(cnt + accept.sum(), total))
		alpha_dot[ind] = of.alpha_dot[accept][:len(ind)].astype(dtype)
		g_aux_all.append(g_aux[accept])
		g_all.append(g[accept])
		cnt += accept.sum()
		if cnt >= total:
			break
		kws['seed'] += 1

	alpha_dot = np.transpose(alpha_dot, (0, -1, 1, 2))
	g_all, g_aux_all = cat_map([g_all, g_aux_all], axis=0)
	g_all, g_aux_all = g_all[:, :total], g_aux_all[:, :total]

	attrs = {
		'f': f,
		'f_aux': f_aux,
		'category': of.category,
		'n_obj': of.n_obj,
		'dim': of.dim,
		'fov': of.fov,
		'res': of.res,
		'z_bg': of.z_bg,
		'obj_r': of.obj_r,
		'obj_bound': of.obj_bound,
		'obj_zlim': of.obj_zlim,
		'vlim_obj': of.vlim_obj,
		'vlim_slf': of.vlim_slf,
		'residual': of.residual,
		'seeds': range(kws['seed'] + 1),
	}
	return alpha_dot, g_all, g_aux_all, attrs


def save_simulation(
		save_dir: str,
		x: np.ndarray,
		g: np.ndarray,
		g_aux: np.ndarray,
		attrs: dict,
		split: dict = None, ):
	n = len(x)
	name = '_'.join([
		f"{attrs['category']}{attrs['n_obj']}",
		f"dim-{attrs['dim']}",
		f"n-{n//1000}k",
	])
	path = pjoin(save_dir, name)
	os.makedirs(path, exist_ok=True)
	# save attrs
	save_obj(
		obj=attrs,
		save_dir=path,
		file_name='attrs',
		verbose=False,
		mode='npy',
	)
	# save data
	split = split if split else {
		'trn': int(0.8 * n),
		'vld': int(0.1 * n),
		'tst': int(0.1 * n),
	}
	assert sum(split.values()) == n
	i = 0
	split_ids = {}
	for k, v in split.items():
		split_ids[k] = range(i, i + v)
		i += v
	for a, b in itertools.combinations(split_ids.values(), 2):
		assert not set(a).intersection(b)

	for lbl, ids in split_ids.items():
		_path = pjoin(path, lbl)
		os.makedirs(_path, exist_ok=True)
		kws = dict(
			save_dir=_path,
			verbose=False,
			mode='npy',
		)
		# generative factors
		kws['obj'] = g[ids]
		kws['file_name'] = 'g'
		save_obj(**kws)
		# generative factors (aux)
		kws['obj'] = g_aux[ids]
		kws['file_name'] = 'g_aux'
		save_obj(**kws)
		# flow frames
		kws['obj'] = x[ids]
		kws['file_name'] = 'x'
		save_obj(**kws)
		# norm
		kws['obj'] = np.sum(sp_lin.norm(
			x[ids], axis=1), axis=(1, 2))
		kws['file_name'] = 'norm'
		save_obj(**kws)
	return


def setup_repeat_data(
		group: h5py.Group,
		kws_hf: dict,
		r_ratio: float = None, ):
	if not group.attrs.get('has_repeats'):
		return None, None, None

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
		r_ratio=r_ratio if r_ratio else
		np.array(group['hf_radius_ratioR']),
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
		'dim': 17, 'sres': 1, 'apply_mask': True}
	radius_ratio = np.array(group['hf_radius_ratio'])
	# inconsistent diameters throughout the expt?
	if len(set(group.attrs.get('diameter'))) != 1:
		if 'hf_radius_ratioR' in group:
			radius_ratio = np.concatenate([
				radius_ratio,
				np.array(group['hf_radius_ratioR']),
			])
		radius_ratio = radius_ratio.mean()
		radius_ratio_r = radius_ratio
	else:
		radius_ratio_r = None

	hf = HyperFlow(
		params=np.array(group['hf_params']),
		center=np.array(group['hf_center']),
		r_ratio=radius_ratio,
		**kws_hf,
	)
	stim = hf.compute_hyperflow(dtype=dtype)
	spks = np.array(group['spks'], dtype=float)
	if 'badspks' in group:
		mask = ~np.array(group['badspks'], dtype=bool)
	else:
		mask = np.ones(len(spks), dtype=bool)
	stim_r, spks_r, good_r = setup_repeat_data(
		group=group,
		kws_hf=kws_hf,
		r_ratio=radius_ratio_r,
	)

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
	combos = [('fixate', i) for i in [0, 1, 2, 4, 8]]
	combos += [('transl', i) for i in [0, 1, 2, 4, 8]]
	combos += [('terrain', i) for i in [1, 2, 4, 8]]
	combos += [('obj', i) for i in [1, 2, 4, 8]]
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
	save_dir = '/home/hadi/Documents/MTMST/data'
	combos = simulation_combos()
	print(f"Simulation combos:\n{combos}")
	pbar = tqdm(combos)
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
