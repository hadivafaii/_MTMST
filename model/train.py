from .utils_model import *
from .dataset import ROFL
from .common import endpoint_error
from .configuration import ConfigTrain
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class BaseTrainer(object):
	def __init__(
			self,
			model: Module,
			cfg: ConfigTrain,
			device: str = None,
			verbose: bool = True,
	):
		super(BaseTrainer, self).__init__()
		self.cfg = cfg
		self.verbose = verbose
		self.device = torch.device(device if device else "cpu")
		self.model = model.to(self.device).eval()

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
		kwargs = dict(
			warmup_iters=len(self.dl_trn) * self.cfg.warmup_epochs,
			n_iters_tot=len(self.dl_trn) * len(epochs)
		)
		if save:
			self.model.create_chkpt_dir(comment)
			self.cfg.save(self.model.chkpt_dir)
			writer = pjoin(
				self.model.cfg.runs_dir,
				os.path.basename(self.model.chkpt_dir),
			)
			self.writer = SummaryWriter(writer)

		pbar = tqdm(epochs)
		for epoch in pbar:
			avg_loss = self.iteration(epoch, **kwargs)
			msg = ', '.join([
				f"epoch # {epoch + 1:d}",
				f"avg loss: {avg_loss:3f}",
			])
			pbar.set_description(msg)
			if self.optim_schedule is not None and epoch > self.cfg.warmup_epochs:
				self.optim_schedule.step()

			if (epoch + 1) % self.cfg.chkpt_freq == 0 and save:
				self.model.save(checkpoint=epoch + 1)

			if (epoch + 1) % self.cfg.eval_freq == 0:
				n_iters = len(self.dl_trn)
				global_step = (epoch + 1) * n_iters
				_ = self.validate(global_step)
				if self.dl_tst is not None:
					_ = self.test(global_step)
		if save:
			pass
			# TODO: save optim and scheduler state dict
		if self.writer is not None:
			self.writer.close()
		return

	def iteration(self, epoch: int = 0, **kwargs):
		raise NotImplementedError

	def validate(self, global_step: int = None):
		raise NotImplementedError

	def test(self, global_step: int = None):
		raise NotImplementedError

	def setup_data(self):
		raise NotImplementedError

	def swap_model(self, new_model, full_setup: bool = True):
		self.model = new_model.to(self.device).eval()
		if full_setup:
			self.setup_data()
			self.setup_optim()

	def setup_optim(self):
		# optimzer
		params = add_weight_decay(
			model=self.model,
			weight_decay=self.cfg.weight_decay,
		)
		if self.cfg.optimizer == 'adamw':
			self.optim = torch.optim.AdamW(
				params=params,
				lr=self.cfg.lr,
				weight_decay=self.cfg.weight_decay,
				betas=(self.cfg.beta1, self.cfg.beta2),
			)
		elif self.cfg.optimizer == 'adamax':
			self.optim = torch.optim.Adamax(
				params=params,
				lr=self.cfg.lr,
				weight_decay=self.cfg.weight_decay,
				betas=(self.cfg.beta1, self.cfg.beta2),
			)
		else:
			raise NotImplementedError(self.cfg.optimizer)

		# scheduler
		if self.cfg.scheduler_type == 'cosine':
			self.optim_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
				self.optim,
				T_max=self.cfg.scheduler_period,
				eta_min=self.cfg.lr_min,
			)
		elif self.cfg.scheduler_type == 'exponential':
			self.optim_schedule = torch.optim.lr_scheduler.ExponentialLR(
				self.optim,
				gamma=self.cfg.scheduler_gamma,
			)
		elif self.cfg.scheduler_type == 'step':
			self.optim_schedule = torch.optim.lr_scheduler.StepLR(
				self.optim,
				step_size=self.cfg.scheduler_period,
				gamma=self.cfg.scheduler_gamma,
			)
		elif self.cfg.scheduler_type == 'cyclic':
			self.optim = torch.optim.SGD(
				self.model.parameters(),
				lr=self.cfg.lr,
				weight_decay=self.cfg.weight_decay,
				momentum=0.9,
			)
			self.optim_schedule = torch.optim.lr_scheduler.CyclicLR(
				self.optim,
				base_lr=self.cfg.lr_min,
				max_lr=self.cfg.lr,
				step_size_up=self.cfg.scheduler_period,
				gamma=self.cfg.scheduler_gamma,
				mode='exp_range',
			)
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


class TrainerVAE(BaseTrainer):
	def __init__(
			self,
			model: Module,
			cfg: ConfigTrain,
			device: str = None,
			**kwargs,
	):
		super(TrainerVAE, self).__init__(
			model=model,
			cfg=cfg,
			device=device,
			**kwargs,
		)

	def iteration(self, epoch: int = 0, **kwargs):
		self.model.train()
		alpha = kl_balancer_coeff(
			groups=self.model.cfg.groups,
			device=self.device,
			fun='square',
		)
		nelbo = AvgrageMeter()
		for i, (x, norm) in enumerate(self.dl_trn):
			global_step = epoch * len(self.dl_trn) + i
			if global_step < kwargs['warmup_iters']:
				lr = self.cfg.lr * float(global_step) / kwargs['warmup_iters']
				for pg in self.optim.param_groups:
					pg['lr'] = lr
			self.optim.zero_grad()
			# forward
			x, norm = self.to([x, norm])
			y, _, log_q, log_p, kl_all, kl_diag = self.model(x)
			if self.cfg.balanced_recon:
				loss_recon = endpoint_error(true=x, pred=y, w=1/norm)
			else:
				loss_recon = endpoint_error(true=x, pred=y)
			beta = kl_coeff(
				step=global_step,
				total_step=self.cfg.kl_anneal_portion * kwargs['n_iters_tot'],
				constant_step=self.cfg.kl_const_portion * kwargs['n_iters_tot'],
				min_kl_coeff=self.cfg.kl_const_coeff,
			)
			balanced_kl, kl_coeffs, kl_vals = kl_balancer(
				kl_all=kl_all,
				coeff=beta,
				alpha=alpha,
			)
			nelbo_batch = loss_recon + balanced_kl
			loss = torch.mean(nelbo_batch)
			if _check_nans(loss, global_step):
				continue
			if self.cfg.lambda_anneal:
				wd_coeff = np.exp(
						beta * np.log(self.cfg.lambda_norm) +
						(1 - beta) * np.log(self.cfg.lambda_init)
				)
			else:
				wd_coeff = self.cfg.lambda_norm
			if self.model.cfg.weight_norm:
				loss_w = self.model.loss_weight()
				loss += wd_coeff * loss_w
			else:
				loss_w = None
			if self.model.cfg.spectral_reg:
				loss_sr = self.model.loss_spectral(self.device)
				loss += wd_coeff * loss_sr
			else:
				loss_sr = None

			loss.backward()
			grad_norm = nn.utils.clip_grad_norm_(
				parameters=self.model.parameters(),
				max_norm=self.cfg.clip_grad,
			)
			self.optim.step()
			nelbo.update(loss.item(), 1)
			# if epoch > self.cfg.warmup_epochs:
			# thres = self.cfg.batch_size * 4
			# if _check_grads(grads, global_step, thres):
			# continue

			if global_step % (len(self.dl_trn) // self.cfg.log_freq) == 0:
				to_write = {
					'train/kl_beta': beta,
					'train/wd_coeff': wd_coeff,
					'train/grad': grad_norm.item(),
					'train/lr': self.optim.param_groups[0]['lr'],
					'train/kl_iter': torch.mean(sum(kl_all)),
					'train/nelbo_avg': nelbo.avg / self.cfg.batch_size,
					'train/loss_tot_iter': loss.item() / self.cfg.batch_size,
					'train/recon_iter': loss_recon.item() / self.cfg.batch_size,
				}
				if self.model.cfg.weight_norm:
					to_write['train/loss_weight_iter'] = loss_w.item()
				if self.model.cfg.spectral_reg:
					to_write['train/loss_spectral_iter']: loss_sr.item()
				total_active = 0
				for j, kl_diag_i in enumerate(kl_diag):
					to_write[f"kl/beta_layer_{j}"] = kl_coeffs[j]
					to_write[f"kl/vals_layer_{j}"] = kl_vals[j]
					n_active = torch.sum(kl_diag_i > 0.1).item()
					to_write[f"kl/active_{j}"] = n_active
					total_active += n_active
				to_write['kl/total_active'] = total_active
				to_write['kl/total_active_ratio'] = total_active /\
					sum(self.model.cfg.groups) / self.model.cfg.n_latent_per_group
				grads = [
					torch.linalg.norm(p.grad).item() for
					n, p in self.model.named_parameters()
				]
				to_write['grads/median'] = np.median(grads)
				to_write['grads/mean'] = np.mean(grads)
				to_write['grads/max'] = np.max(grads)
				for k, v in to_write.items():
					self.writer.add_scalar(k, v, global_step)

		return nelbo.avg / self.cfg.batch_size

	def validate(self, global_step: int = None):
		pass

	def test(self, global_step: int = None):
		pass

	def setup_data(self):
		# create datasets
		f = pjoin(self.model.cfg.data_dir, self.model.cfg.h_pre)
		f = h5py.File(f, mode='r')
		ds_trn = ROFL(np.array(f['tst']['x'], dtype=float))
		ds_vld = ROFL(np.array(f['vld']['x'], dtype=float))
		ds_tst = ROFL(np.array(f['tst']['x'], dtype=float))
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


def _check_nans(loss, global_step: int):
	if torch.isnan(loss).sum().item():
		msg = 'nan encountered in loss. '
		msg += 'optimizer will detect this & skip. '
		msg += f"global step = {global_step}"
		print(msg)
		return True


def _check_grads(grads, global_step: int, thres=1e3):
	if np.mean(grads) > thres:
		msg = 'exploding gradients encountered. '
		msg += 'optimizer will detect this & skip. '
		msg += f"(global step = {global_step}):\n"
		msg += f"mean(grads) = {np.mean(grads):0.1g},  "
		msg += f"median(grads) = {np.median(grads):0.1g},  "
		msg += f"max(grads) = {np.max(grads):0.1g},  "
		print(msg)
		return True
