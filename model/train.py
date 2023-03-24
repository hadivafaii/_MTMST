import torch.optim

from figures.fighelper import *
from .dataset import ROFL
from analysis.regression import regress
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class _BaseTrainer(object):
	def __init__(
			self,
			model: Module,
			cfg: ConfigTrain,
			device: str = 'cpu',
			verbose: bool = False,
	):
		super(_BaseTrainer, self).__init__()
		self.cfg = cfg
		self.verbose = verbose
		self.device = torch.device(device)
		self.model = model.to(self.device).eval()
		self.n_iters = None

		self.writer = None
		self.dl_trn = None
		self.dl_vld = None
		self.dl_tst = None
		self.setup_data()

		self.optim = None
		self.optim_schedule = None
		self.setup_optim()

		if self.verbose:
			tot = sum([
				p.nelement() for p in
				self.model.parameters()
			])
			if tot // 1e6 > 0:
				tot = f"{np.round(tot / 1e6, 2):1.1f} M"
			elif tot // 1e3 > 0:
				tot = f"{np.round(tot / 1e3, 2):1.1f} K"
			print(f"\n# params: {tot}")

	def train(
			self,
			epochs: Union[int, range] = None,
			comment: str = None,
			save: bool = True, ):
		epochs = epochs if epochs else self.cfg.epochs
		assert isinstance(epochs, (int, range)), "allowed: {int, range}"
		epochs = range(epochs) if isinstance(epochs, int) else epochs
		comment if comment else self.cfg.name()
		kwargs = dict(n_iters_warmup=int(np.round(
			self.n_iters * self.cfg.warmup_portion)))
		if save:
			self.model.create_chkpt_dir(comment)
			self.cfg.save(self.model.chkpt_dir)
			writer = pjoin(
				self.model.cfg.runs_dir,
				os.path.basename(self.model.chkpt_dir),
			)
			self.writer = SummaryWriter(writer)
		if self.cfg.scheduler_type == 'cosine':
			self.optim_schedule.T_max *= len(self.dl_trn)
		else:
			raise NotImplementedError

		pbar = tqdm(epochs)
		for epoch in pbar:
			avg_loss = self.iteration(epoch, **kwargs)
			msg = ', '.join([
				f"epoch # {epoch + 1:d}",
				f"avg loss: {avg_loss:3f}",
			])
			pbar.set_description(msg)
			if not save:
				continue
			if (epoch + 1) % self.cfg.chkpt_freq == 0:
				self.save(
					checkpoint=epoch + 1,
					path=self.model.chkpt_dir,
				)
			if (epoch + 1) % self.cfg.eval_freq == 0:
				gstep = (epoch + 1) * len(self.dl_trn)
				_ = self.validate(gstep)
		if self.writer is not None:
			self.writer.close()
		return

	def iteration(self, epoch: int = 0, **kwargs):
		raise NotImplementedError

	def validate(self, epoch: int = None):
		raise NotImplementedError

	def setup_data(self):
		raise NotImplementedError

	def swap_model(self, new_model, full_setup: bool = True):
		self.model = new_model.to(self.device).eval()
		if full_setup:
			self.setup_data()
			self.setup_optim()

	def save(self, checkpoint: int, path: str):
		metadata = {
			'checkpoint': checkpoint,
			'global_step': checkpoint * len(self.dl_trn),
		}
		state_dict = {
			'metadata': metadata,
			'model': self.model.state_dict(),
			'optim': self.optim.state_dict(),
		}
		if self.optim_schedule is not None:
			state_dict['scheduler'] = self.optim_schedule.state_dict()
		fname = '-'.join([
			'+'.join([
				type(self.model).__name__,
				type(self).__name__]),
			f"{checkpoint:04d}",
			f"({now(True)}).pt",
		])
		fname = pjoin(path, fname)
		torch.save(state_dict, fname)
		return

	def setup_optim(self):
		# optimzer
		params = self.model.parameters()
		kws = dict(
			params=params,
			lr=self.cfg.lr,
			**self.cfg.optimizer_kws,
		)
		if self.cfg.optimizer == 'adamax':
			self.optim = torch.optim.Adamax(**kws)
		elif self.cfg.optimizer == 'adam':
			self.optim = torch.optim.Adam(**kws)
		elif self.cfg.optimizer == 'adamw':
			self.optim = torch.optim.AdamW(**kws)
		elif self.cfg.optimizer == 'radam':
			self.optim = torch.optim.RAdam(**kws)
		else:
			raise NotImplementedError(self.cfg.optimizer)

		# scheduler
		if self.cfg.scheduler_type == 'cosine':
			self.optim_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
				self.optim, **self.cfg.scheduler_kws)
		elif self.cfg.scheduler_type == 'exponential':
			self.optim_schedule = torch.optim.lr_scheduler.ExponentialLR(
				self.optim, **self.cfg.scheduler_kws)
		elif self.cfg.scheduler_type == 'step':
			self.optim_schedule = torch.optim.lr_scheduler.StepLR(
				self.optim, **self.cfg.scheduler_kws)
		elif self.cfg.scheduler_type == 'cyclic':
			self.optim = torch.optim.SGD(
				params=params,
				lr=self.cfg.lr,
				momentum=0.9,
				weight_decay=self.cfg.optimizer_kws.get('weight_decay', 0),
			)
			self.optim_schedule = torch.optim.lr_scheduler.CyclicLR(
				self.optim, **self.cfg.scheduler_kws)
		elif self.cfg.scheduler_type is None:
			self.optim_schedule = None
		else:
			raise NotImplementedError(self.cfg.scheduler_type)
		return

	def to(self, x, dtype=torch.float32) -> Union[torch.Tensor, List[torch.Tensor]]:
		kws = dict(device=self.device, dtype=dtype)
		if isinstance(x, (tuple, list)):
			return [
				e.to(**kws) if torch.is_tensor(e)
				else torch.tensor(e, **kws)
				for e in x
			]
		else:
			if torch.is_tensor(x):
				return x.to(**kws)
			else:
				return torch.tensor(x, **kws)


class TrainerVAE(_BaseTrainer):
	def __init__(
			self,
			model: Module,
			cfg: ConfigTrain,
			**kwargs,
	):
		super(TrainerVAE, self).__init__(
			model=model, cfg=cfg, **kwargs)
		self.n_iters = self.cfg.epochs * len(self.dl_trn)
		if self.cfg.kl_balancer is not None:
			alphas = kl_balancer_coeff(
				groups=self.model.cfg.groups,
				fun=self.cfg.kl_balancer,
			)
			self.alphas = self.to(alphas)
		else:
			self.alphas = None
		if self.cfg.kl_anneal_cycles == 0:
			self.betas = beta_anneal_linear(
				n_iters=self.n_iters,
				beta=self.cfg.kl_beta,
				anneal_portion=self.cfg.kl_anneal_portion,
				constant_portion=self.cfg.kl_const_portion,
				min_beta=self.cfg.kl_beta_min,
			)
		else:
			betas = beta_anneal_cosine(
				n_iters=self.n_iters,
				n_cycles=self.cfg.kl_anneal_cycles,
				portion=self.cfg.kl_anneal_portion,
				start=np.arccos(
					1 - 2 * self.cfg.kl_beta_min
					/ self.cfg.kl_beta) / np.pi,
				beta=self.cfg.kl_beta,
			)
			beta_cte = int(np.round(self.cfg.kl_const_portion * self.n_iters))
			beta_cte = np.ones(beta_cte) * self.cfg.kl_beta_min
			self.betas = np.insert(betas, 0, beta_cte)[:self.n_iters]
		if self.cfg.lambda_anneal:
			self.wd_coeffs = beta_anneal_linear(
				n_iters=self.n_iters,
				beta=self.cfg.lambda_norm,
				anneal_portion=self.cfg.kl_anneal_portion,
				constant_portion=self.cfg.kl_const_portion,
				min_beta=self.cfg.lambda_init,
			)
		else:
			self.wd_coeffs = np.ones(self.n_iters)
			self.wd_coeffs *= self.cfg.lambda_norm

	def iteration(self, epoch: int = 0, **kwargs):
		self.model.train()
		nelbo = AvgrageMeter()
		for i, (x, norm) in enumerate(self.dl_trn):
			gstep = epoch * len(self.dl_trn) + i
			if gstep < kwargs['n_iters_warmup']:
				lr = self.cfg.lr * gstep / kwargs['n_iters_warmup']
				for pg in self.optim.param_groups:
					pg['lr'] = lr
			self.optim.zero_grad()
			# send to device
			if x.device != self.device:
				x, norm = self.to([x, norm])
			# forward
			y, _, q, p = self.model(x)
			# loss
			l1, l2, epe, cos = self.model.loss_recon(
				x=x, y=y, w=1 / norm)
			loss_recon = l1 + l2 + epe + cos.mul(0.1)
			kl_all, kl_diag = self.model.loss_kl(q, p)
			# balance kl
			balanced_kl, kl_coeffs, kl_vals = kl_balancer(
				kl_all=kl_all,
				alpha=self.alphas,
				coeff=self.betas[gstep],
				beta=self.cfg.kl_beta,
			)
			nelbo_batch = loss_recon + balanced_kl
			loss = torch.mean(nelbo_batch)
			# add regularization
			cond_reg_weight = self.cfg.lambda_norm > 0
			if cond_reg_weight:
				loss_w = self.model.loss_weight()
				loss += self.wd_coeffs[gstep] * loss_w
			else:
				loss_w = None
			cond_reg_spectral = self.cfg.lambda_norm > 0 \
				and self.cfg.spectral_reg and \
				not self.model.cfg.spectral_norm
			if cond_reg_spectral:
				loss_sr = self.model.loss_spectral(
					device=self.device, name='w')
				loss += self.wd_coeffs[gstep] * loss_sr
			else:
				loss_sr = None
			# backward
			loss.backward()
			nelbo.update(loss.item(), 1)
			if self.cfg.grad_clip is not None:
				grad_norm = nn.utils.clip_grad_norm_(
					parameters=self.model.parameters(),
					max_norm=self.cfg.grad_clip,
				).item()
			else:
				grad_norm = None
			self.writer.add_scalar('grads/loss_before_skip', loss, gstep)
			self.writer.add_scalar('grads/norm_before_skip', grad_norm, gstep)
			# if not _check_nans(loss, gstep):
			# if grad_norm > 1e4:
			# print(f'gstep={gstep}, grad norm = {grad_norm} . . . skipped')
			# continue
			self.optim.step()
			cond_schedule = (
				gstep > kwargs['n_iters_warmup']
				and self.optim_schedule is not None
			)
			if cond_schedule:
				self.optim_schedule.step()
			# write
			cond_write = (
				self.writer is not None and
				gstep % self.cfg.log_freq == 0
				and gstep > 0
			)
			if not cond_write:
				continue
			to_write = {
				'train/beta': self.betas[gstep],
				'train/reg_coeff': self.wd_coeffs[gstep],
				'train/lr': self.optim.param_groups[0]['lr'],
				'train/loss_kl': torch.mean(sum(kl_all)).item(),
				'train/loss_recon': torch.mean(loss_recon).item(),
				'train/loss_recon_l1': torch.mean(l1).item(),
				'train/loss_recon_l2': torch.mean(l2).item(),
				'train/loss_recon_epe': torch.mean(epe).item(),
				'train/loss_recon_cos': torch.mean(cos).item(),
				'train/loss_tot': loss.item(),
				'train/nelbo_avg': nelbo.avg,
			}
			if cond_reg_weight:
				to_write['train/reg_weight'] = loss_w.item()
			if cond_reg_spectral:
				to_write['train/reg_spectral'] = loss_sr.item()
			total_active = 0
			for j, kl_diag_i in enumerate(kl_diag):
				to_write[f"kl_full/beta_layer_{j}"] = kl_coeffs[j].item()
				to_write[f"kl_full/vals_layer_{j}"] = kl_vals[j].item()
				n_active = torch.sum(kl_diag_i > 0.1).item()
				to_write[f"kl_full/active_{j}"] = n_active
				total_active += n_active
			to_write['kl/total_active'] = total_active
			ratio = total_active / self.model.cfg.total_latents()
			to_write['kl/total_active_ratio'] = ratio
			grads = [
				torch.linalg.norm(p.grad).item()
				for p in self.model.parameters()
			]
			to_write['grads/median'] = np.median(grads)
			to_write['grads/mean'] = np.mean(grads)
			to_write['grads/max'] = np.max(grads)
			if grad_norm is not None:
				to_write['grads/norm'] = grad_norm
			for k, v in to_write.items():
				self.writer.add_scalar(k, v, gstep)

		return nelbo.avg

	def validate(
			self,
			gstep: int = None,
			n_samples: int = 4096, ):
		self.model.eval()
		data, loss = self.forward('vld')
		# sample? plot?
		if gstep is not None:
			i = int(gstep / len(self.dl_trn))
			cond = i % (self.cfg.eval_freq * 5) == 0
		else:
			cond = True
		cond = cond and n_samples is not None
		if cond:
			x_sample, z_sample, regr, figs = self.plot(
				n_samples=n_samples)
			data = {
				'x_sample': x_sample,
				'z_sample': z_sample,
				**regr,
				**figs,
			}
		else:
			regr, figs = None, None
		# write
		if gstep is not None:
			for k, v in loss.items():
				self.writer.add_scalar(f"eval/{k}", v.mean(), gstep)
			if cond:
				r = np.diag(regr['regr/r']).mean()
				mi = np.max(regr['regr/mi'], axis=0).mean()
				self.writer.add_scalar(f"eval/r", r, gstep)
				self.writer.add_scalar(f"eval/mi", mi, gstep)
				for k, v in figs.items():
					self.writer.add_figure(k, v, gstep)
		return data, loss

	def forward(self, dl: str, loss: bool = True):
		assert dl in ['trn', 'vld', 'tst']
		dl = getattr(self, f"dl_{dl}")
		if dl is None:
			return
		self.model.eval()

		x_all, y_all, z_all = [], [], []
		l1, l2, epe, cos, kl = [], [], [], [], []
		for i, (x, norm) in enumerate(dl):
			x, norm = self.to([x, norm])
			with torch.no_grad():
				y, z, q, p = self.model(x)
			# loss
			if loss:
				_l1, _l2, _epe, _cos = self.model.loss_recon(
					x=x, y=y, w=1 / norm)
				l1.append(to_np(_l1))
				l2.append(to_np(_l2))
				epe.append(to_np(_epe))
				cos.append(to_np(_cos))
				kl_all, _ = self.model.loss_kl(q, p)
				kl.append(to_np(sum(kl_all)))
			# data
			x_all.append(to_np(x))
			y_all.append(to_np(y))
			z_all.append(to_np(z))

		x, y, z = cat_map([x_all, y_all, z_all])
		data = {
			'x': x,
			'y': y,
			'z': z,
		}
		if loss:
			l1, l2, epe, cos, kl = cat_map(
				[l1, l2, epe, cos, kl])
			loss = {
				'kl': kl,
				'l1': l1,
				'l2': l2,
				'epe': epe,
				'cos': cos,
			}
		return data, loss

	def sample(self, n_samples: int = 4096, t: float = 1.0):
		num = n_samples / self.cfg.batch_size
		num = int(np.ceil(num))
		x_sample, z_sample = [], []
		tot = 0
		for _ in range(num):
			n = self.cfg.batch_size
			if tot + self.cfg.batch_size > n_samples:
				n = n_samples - tot
			with torch.no_grad():
				_x, _z, _ = self.model.sample(
					n=n, t=t, device=self.device)
			x_sample.append(to_np(_x))
			z_sample.append(to_np(_z))
			tot += self.cfg.batch_size
		x_sample, z_sample = cat_map([x_sample, z_sample])
		return x_sample, z_sample

	def regress(self, n_fwd: int = 10):
		z_vld, z_tst = [], []
		for _ in range(n_fwd):
			z_vld.append(np.expand_dims(
				self.forward('vld', False)[0]['z'], 0))
			z_tst.append(np.expand_dims(
				self.forward('tst', False)[0]['z'], 0))
		z_vld, z_tst = cat_map([z_vld, z_tst])
		z_vld = z_vld.mean(0)
		z_tst = z_tst.mean(0)
		g_vld = self.dl_vld.dataset.factors
		g_tst = self.dl_tst.dataset.factors
		mi, r, lr = regress(z_vld, g_vld, z_tst, g_tst)
		output = {
			'z_vld': z_vld,
			'z_tst': z_tst,
			'g_vld': g_vld,
			'g_tst': g_tst,
			'regr/mi': mi,
			'regr/r': r,
			'regr/lr': lr,
		}
		return output

	def plot(self, sample: dict = None, regr: dict = None, **kwargs):
		regr = regr if regr else self.regress(
			**filter_kwargs(self.regress, kwargs))
		if sample is None:
			x_sample, z_sample = self.sample(
				**filter_kwargs(self.sample, kwargs))
		else:
			x_sample, z_sample = sample['x'], sample['z']

		figs = {}
		# samples (opticflow)
		fig, _ = show_opticflow(
			x_sample, n=6, display=False)
		figs['fig/sample'] = fig

		# corr (latents)
		r = 1 - sp_dist.pdist(
			regr['z_vld'].T, 'correlation')
		r = sp_dist.squareform(r)
		fig, _ = show_heatmap(r, annot=False, display=False)
		figs['fig/corr_z'] = fig

		# corr (regression)
		names = self.dl_tst.dataset.factor_names
		_tx = [f"({i:02d})" for i in range(len(names))]
		_ty = [f"{e} ({i:02d})" for i, e in names.items()]
		rd = np.diag(regr['regr/r'])
		title = f"all  =  {rd.mean():0.3f} ± {rd.std():0.3f}  "
		title += r'$(\mu \pm \sigma)$' + '\n'
		name_groups = collections.defaultdict(list)
		for i, lbl in names.items():
			k = '_'.join(lbl.split('_')[:-1])
			name_groups[k].append(i)
		for k, ids in name_groups.items():
			title += f"{k} :  {rd[ids].mean():0.2f},"
			title += ' ' * 5
		fig, _ = show_heatmap(
			r=regr['regr/r'],
			title=title,
			cmap='PiYG',
			xticklabels=_tx,
			yticklabels=_ty,
			annot_kws={'fontsize': 12},
			figsize=(10, 8),
			display=False,
		)
		figs['fig/regression'] = fig

		# mutual info
		fig, _ = show_heatmap(
			r=regr['regr/mi'],
			yticklabels=_ty,
			title=f"{self.model.cfg.name()}",
			tick_labelsize_y=7,
			title_fontsize=15,
			title_y=1.02,
			vmin=0,
			vmax=0.65,
			cmap='rocket',
			linecolor='dimgrey',
			cbar_kws={'pad': 0.001, 'shrink': 0.7},
			figsize=(20, 2.5),
			annot=False,
			display=False,
		)
		figs['fig/mutual_info'] = fig

		# TODO: do mutual info normalized by scales plot
		# scales, level_ids = self.model.latent_scales()

		# hist (latents)
		# fig, _ = plot_latents_hist(
		# 	z=z_sample,
		# 	scales=scales,
		# 	display=False,
		# )
		# figs['fig/hist_z'] = fig

		# hist (samples)
		# fig, _ = plot_opticflow_hist(
		# 	x=x_sample, display=False)
		# figs['fig/hist_x_sample'] = fig
		return x_sample, z_sample, regr, figs

	def setup_data(self, gpu: bool = True):
		# create datasets
		f = h5py.File(self.model.cfg.h_pre)
		device = self.device if gpu else None
		ds_trn = ROFL(f['trn'], device=device)
		ds_vld = ROFL(f['vld'], device=device)
		ds_tst = ROFL(f['tst'], device=device)
		f.close()
		# cleate dataloaders
		kws = dict(
			batch_size=self.cfg.batch_size,
			drop_last=True,
			shuffle=True,
		)
		self.dl_trn = DataLoader(ds_trn, **kws)
		kws.update({'drop_last': False, 'shuffle': False})
		self.dl_vld = DataLoader(ds_vld, **kws)
		self.dl_tst = DataLoader(ds_tst, **kws)
		return


def _check_grads(
		grads: List[float],
		thres: float,
		gstep: int,
		fn: Callable = np.max, ):
	if fn(grads) > thres:
		msg = 'diverging grad encountered:\n'
		msg += f'fn: {fn.__name__}, grad: {fn(grads):0.1f} > {thres}'
		msg += f'\nglobal step = {gstep}, skipping . . . '
		print(msg)
		return True


def _check_nans(loss, gstep: int):
	if torch.isnan(loss).sum().item():
		msg = 'nan encountered in loss. '
		msg += 'optimizer will detect this & skip. '
		msg += f"global step = {gstep}"
		print(msg)
		return True
	else:
		return False
