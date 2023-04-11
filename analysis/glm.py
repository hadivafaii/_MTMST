from .helper import *
from .linear import compute_sta
from vae.train_vae import TrainerVAE
from readout.readout import process_ftrs
from analysis.opticflow import HyperFlow
from base.dataset import setup_repeat_data


class ReadoutGLM(object):
	def __init__(
			self,
			expt: str,
			tr: TrainerVAE,
			n_pcs: int = 500,
			n_lags: int = 19,
			dtype: str = 'float32',
			verbose: bool = False,
	):
		super(ReadoutGLM, self).__init__()
		self.expt = expt
		self.tr = tr
		self.n_pcs = n_pcs
		self.n_lags = n_lags
		self.verbose = verbose
		self.dtype = dtype
		self.pcs = {}
		self.x = None
		self.x_r = None

	def fit(self, **kwargs):
		s, s_r = self._load_neuron(**kwargs)
		e, e_r = self._xtract(s, s_r, **kwargs)
		self._sta(e)
		self._best_lags_pix()
		self._pca(e, e_r)
		return

	def _load_neuron(self, **kwargs):
		kws_hf = dict(size=19, sres=1, radius=6)
		kws_hf = setup_kwargs(kws_hf, kwargs)
		stim, spks, good, stim_r, spks_r, good_r = prepare_data(
			expt=self.expt, tr=self.tr, kws_hf=kws_hf, rescale=2.25)
		self.nc = spks.shape[1]
		self.spks = spks
		self.good = good
		self.spks_r = spks_r
		self.good_r = good_r
		if self.verbose:
			print('[PROGRESS] neural data loaded')
		return stim, stim_r

	def _xtract(self, stim, stim_r, normalize=True, **kwargs):
		kws_process = dict(scale=4, pool='max', act_fn='swish')
		kws_process = setup_kwargs(kws_process, kwargs)
		enc = push(self.tr, stim, kws_process, verbose=self.verbose)
		enc_r = push(self.tr, stim_r, kws_process, verbose=self.verbose)
		if normalize:
			mu, sd = enc.mean(), enc.std()
			enc = (enc - mu) / sd
			if enc_r is not None:
				enc_r = (enc_r - mu) / sd
		if self.verbose:
			print('[PROGRESS] features extracted')
		return enc, enc_r

	def _sta(self, enc: np.ndarray):
		self.sta = compute_sta(
			stim=enc,
			good=self.good,
			spks=self.spks,
			lags=self.n_lags,
			verbose=self.verbose,
			zscore=True,
		)
		if self.verbose:
			print('[PROGRESS] sta computed')
		return

	def _best_lags_pix(self):
		# best lags
		dims = (2, 3, 4)
		self.temporal = np.mean(self.sta ** 2, axis=dims)
		self.best_lags = np.argmax(self.temporal[:, ::-1], axis=1)
		# best pix at best lags
		self.best_ij = np.zeros((self.nc, 2), dtype=int)
		self.spatial = np.zeros((self.nc, *self.sta.shape[-2:]))
		for idx, lag in enumerate(self.best_lags):
			norm = np.mean(self.sta[idx][lag] ** 2, axis=0)
			i, j = np.unravel_index(np.argmax(norm), norm.shape)
			self.best_ij[idx] = i, j
			self.spatial[idx] = norm
		if self.verbose:
			print('[PROGRESS] best lag/pix estimated')
		return

	def _pca(self, enc, enc_r):
		x = np.empty((self.nc, len(enc), self.n_pcs))
		if enc_r is not None:
			x_r = np.empty((self.nc, len(enc_r), self.n_pcs))
		else:
			x_r = None
		for idx in range(self.nc):
			pc = sk_decomp.PCA(
				n_components=self.n_pcs,
				svd_solver='full',
			)
			i, j = self.best_ij[idx]
			x[idx] = pc.fit_transform(enc[..., i, j])
			if enc_r is not None:
				x_r[idx] = pc.transform(enc_r[..., i, j])
			self.pcs[idx] = pc
		self.x = x
		self.x_r = x_r
		if self.verbose:
			print('[PROGRESS] fitting PCA done')
		return


def push(
		tr: TrainerVAE,
		stim: np.ndarray,
		kws_process: dict,
		which: str = 'enc',
		verbose: bool = False,
		use_ema: bool = False,
		dtype: str = 'float32', ):
	if stim is None:
		return
	# feature sizes
	assert kws_process['pool'] != 'none'
	s = kws_process['scale']
	model = tr.select_model(use_ema)
	n_ftrs_enc, n_ftrs_dec = model.ftr_sizes()
	nf_enc = sum(n_ftrs_enc.values())
	nf_dec = sum(n_ftrs_dec.values())
	if which == 'enc':
		nf = nf_enc
	elif which == 'dec':
		nf = nf_dec
	elif which == 'both':
		nf = nf_enc + nf_dec
	else:
		raise NotImplementedError(which)

	shape = (len(stim), nf, s, s)
	x = np.empty(shape, dtype=dtype)
	n_iter = len(x) / tr.cfg.batch_size
	n_iter = int(np.ceil(n_iter))
	for i in tqdm(range(n_iter), disable=not verbose):
		a = i * tr.cfg.batch_size
		b = min(a + tr.cfg.batch_size, len(x))
		*_, ftr = model.xtract_ftr(
			x=tr.to(stim[a:b]), full=True)
		ftr = process_ftrs(ftr[which], **kws_process)
		ftr = torch.cat(list(ftr.values()), dim=1)
		x[a:b] = to_np(ftr).astype(dtype)
	return x


def prepare_data(
		expt: str,
		tr: TrainerVAE,
		kws_hf: dict = None,
		rescale: float = 2.25,
		dtype: str = 'float32', ):
	file = h5py.File(tr.model.cfg.h_file, 'r')
	grp = file[expt]

	kws_hf = kws_hf if kws_hf else {
		'size': 65,
		'sres': 1,
		'radius': 16,
	}
	hf = HyperFlow(
		params=np.array(grp['hyperflow'])[:, 2:],
		center=np.array(grp['hyperflow'])[:, :2],
		**kws_hf,
	)
	stim = hf.compute_hyperflow(dtype=dtype)
	spks = np.array(grp['spks'], dtype=float)
	good = ~np.array(grp['badspks'])
	good = np.where(good)[0]
	stim_r, spks_r, good_r = setup_repeat_data(
		group=grp, kws_hf=kws_hf, use_hf=True)
	file.close()

	stim_scale = np.max(np.abs(stim))
	stim *= rescale / stim_scale
	if stim_r is not None:
		stim_r *= rescale / stim_scale

	return stim, spks, good, stim_r, spks_r, good_r
