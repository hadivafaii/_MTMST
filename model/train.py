from .utils_model import *
from .configuration import ConfigTrain
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR, CyclicLR
from torch.optim import AdamW, Adamax, SGD


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
			n_params = sum([
				p.nelement() for p in
				self.model.parameters()
			])
			print(f"\nTotal Parameters: {n_params}")

	def train(
			self,
			epochs: Union[int, range],
			comment: str = None,
			save: bool = True, ):
		assert isinstance(epochs, (int, range)), "must be either int or range"
		epochs = range(epochs) if isinstance(epochs, int) else epochs

		if save:
			self.model.create_chkpt_dir(comment)
			writer = pjoin(
				self.model.cfg.runs_dir,
				os.path.basename(self.model.chkpt_dir),
			)
			self.writer = SummaryWriter(writer)

		pbar = tqdm(epochs)
		for epoch in pbar:
			avg_loss = self.iteration(epoch)
			msg = ', '.join([
				f"epoch # {epoch + 1:d}",
				f"avg loss: {avg_loss:3f}",
			])
			pbar.set_description(msg)
			if self.optim_schedule is not None:
				self.optim_schedule.step()

			if (epoch + 1) % self.cfg.chkpt_freq == 0 and save:
				self.model.save(checkpoint=epoch + 1)

			if (epoch + 1) % self.cfg.eval_freq == 0:
				n_iters = len(self.dl_trn)
				global_step = (epoch + 1) * n_iters
				_ = self.validate(global_step)
				if self.dl_tst is not None:
					_ = self.test(global_step)

	def iteration(self, epoch: int = 0):
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
		params = add_weight_decay(
			model=self.model,
			weight_decay=self.cfg.weight_decay,
		)
		if self.cfg.optimizer == 'adamw':
			self.optim = AdamW(
				params=params,
				lr=self.cfg.lr,
				weight_decay=self.cfg.weight_decay,
				betas=(self.cfg.beta1, self.cfg.beta2),
			)
		elif self.cfg.optimizer == 'adamax':
			self.optim = Adamax(
				params=params,
				lr=self.cfg.lr,
				weight_decay=self.cfg.weight_decay,
				betas=(self.cfg.beta1, self.cfg.beta2),
			)
		else:
			raise NotImplementedError(self.cfg.optimizer)

		if self.cfg.scheduler_type == 'cosine':
			self.optim_schedule = CosineAnnealingLR(
				self.optim,
				T_max=self.cfg.scheduler_period,
				eta_min=self.cfg.lr_min,
			)
		elif self.cfg.scheduler_type == 'exponential':
			self.optim_schedule = ExponentialLR(
				self.optim,
				gamma=self.cfg.scheduler_gamma,
			)
		elif self.cfg.scheduler_type == 'step':
			self.optim_schedule = StepLR(
				self.optim,
				step_size=self.cfg.scheduler_period,
				gamma=self.cfg.scheduler_gamma,
			)
		elif self.cfg.scheduler_type == 'cyclic':
			self.optim = SGD(
				self.model.parameters(),
				lr=self.cfg.lr,
				weight_decay=self.cfg.weight_decay,
				momentum=0.9,
			)
			self.optim_schedule = CyclicLR(
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

	def to(self, x, dtype=torch.float32) -> Union[torch.Tensor, List[torch.Tensor, ...]]:
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

	def iteration(self, epoch: int = 0):
		self.model.train()
		alpha = kl_balancer_coeff(
			groups=self.model.cfg.groups,
			device=self.device,
			fun='square',
		)
		nelbo = AvgrageMeter()
		pass


def _check_for_nans(loss, global_step: int):
	if torch.isnan(loss).sum().item():
		msg = 'nan encountered in loss. '
		msg += 'optimizer will detect this & skip. '
		msg += f"global step = {global_step}"
		raise RuntimeWarning(msg)
