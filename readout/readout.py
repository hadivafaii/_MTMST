from base.common import *


class _Readout(Module):
	def __init__(
			self,
			cfg: ConfigReadout,
			n_ftrs: int = None,
			**kwargs,
	):
		super(_Readout, self).__init__(cfg, **kwargs)
		self.relu = get_act_fn('relu')
		self.dropout = nn.Dropout(self.cfg.dropout)
		self.scale = min(list(self.cfg.n_ftrs))
		if self.cfg.act_fn == 'learned_softplus':
			self.act_fn = LearnedSoftPlus()
		else:
			self.act_fn = get_act_fn(self.cfg.act_fn)
		self.criterion = nn.PoissonNLLLoss(
			log_input=True if self.act_fn is None else False,
			reduction='none',
		)
		if n_ftrs is None:
			n_ftrs = sum(self.cfg.n_ftrs.values())
		self.fc = nn.Linear(n_ftrs, 1)

	def forward(self, ftr: dict):
		raise NotImplementedError

	def loss(self, pred, tgt, mask):
		loss = self.criterion(pred, tgt)
		loss = torch.sum(loss[mask == 1])
		loss.div_(mask.sum())
		return loss

	def process(self, ftr: dict):
		ftr = process_ftrs(
			ftr=ftr,
			scale=self.scale,
			pool=self.cfg.pool,
			act_fn='swish',
		)
		if self.cfg.pool != 'none':
			ftr = torch.cat(list(ftr.values()), dim=1)
		return ftr


class ReadoutSep(_Readout):
	def __init__(self, cfg: ConfigReadout, **kwargs):
		nf = sum(cfg.n_ftrs.values()) * cfg.n_tkers
		super(ReadoutSep, self).__init__(cfg, nf, **kwargs)
		self._init()
		if self.verbose:
			self.print()

	def forward(self, ftr: dict):
		ftr = self.process(ftr)
		if self.cfg.pool == 'none':
			out = []
			for s, x in ftr.items():
				m = self.sker[str(s)]
				out.append(self._op(x, m))
			out = torch.cat(out, dim=1)
		else:
			out = self._op(ftr, self.sker)
		out = self.relu(out)
		out = self.dropout(out)
		out = self.fc(out)
		if self.act_fn is not None:
			out = self.act_fn(out)
		return out

	def loss_weight(self):
		return self.l1_weight(
			torch.cat(self.all_log_norm), self.w_tgt)

	def _op(self, x: torch.Tensor, m: nn.Module):
		tau, nf, w, h = x.size()		# time, nf, s, s
		x = m(x.reshape(-1, w * h))		# time * nf, n_skers
		x = x.view(tau, nf, -1)			# time, nf, n_skers
		x = torch.movedim(x, 0, -1)		# nf, n_skers, time
		x = self.tker(x)				# nf, n_tkers, time
		x = x.view(-1, tau).T			# time, nf * n_tkers
		return x.contiguous()

	def _init(self):
		# tker
		kws = dict(
			in_channels=self.cfg.n_skers,
			out_channels=self.cfg.n_tkers,
			kernel_size=self.cfg.n_lags,
		)
		self.tker = Conv1D(**kws)
		# sker
		kws = dict(out_features=self.cfg.n_skers)
		if self.cfg.pool == 'none':
			self.sker = nn.ModuleDict()
			for s in self.cfg.n_ftrs:
				kws['in_features'] = s ** 2
				self.sker[str(s)] = Linear(**kws)
		else:
			kws['in_features'] = self.scale ** 2
			self.sker = Linear(**kws)
		# weight loss
		self.all_log_norm = []
		for m in self.modules():
			if hasattr(m, 'lognorm'):
				self.all_log_norm.append(m.lognorm)
		w_tgt = torch.zeros_like(torch.cat(self.all_log_norm))
		self.register_buffer('w_tgt', w_tgt)
		self.l1_weight = nn.SmoothL1Loss(
			beta=0.1, reduction='mean')
		return


class Readout3D(_Readout):
	def __init__(self, cfg: ConfigReadout, **kwargs):
		super(Readout3D, self).__init__(cfg, **kwargs)
		self._init()
		if self.verbose:
			self.print()

	def forward(self, ftr: dict):
		ftr = self.process(ftr)
		if self.cfg.pool == 'none':
			out = []
			for s, x in ftr.items():
				m = self.kernel[str(s)]
				out.append(self._op(x, m))
			out = torch.cat(out, dim=1)
		else:
			out = self._op(ftr, self.kernel)
		out = self.relu(out)
		out = self.dropout(out)
		out = self.fc(out)
		if self.act_fn is not None:
			out = self.act_fn(out)
		return out

	def _op(self, x: torch.Tensor, m: nn.Module):
		x = torch.movedim(x, 0, -1).unsqueeze(1)
		x = m(x).squeeze()[..., :-self.pad].T
		return x.contiguous()

	def _init(self):
		self.pad = self.cfg.n_lags - 1
		kws = dict(
			in_channels=1,
			out_channels=1,
			padding=(0, 0, self.pad),
		)
		if self.cfg.pool == 'none':
			self.kernel = nn.ModuleDict()
			for s in self.cfg.n_ftrs:
				kws['kernel_size'] = (s, s, self.cfg.n_lags)
				self.kernel[str(s)] = nn.Conv3d(**kws)
		else:
			kws['kernel_size'] = (
				self.scale,
				self.scale,
				self.cfg.n_lags,
			)
			self.kernel = nn.Conv3d(**kws)
		fn = AddNorm('weight', nn.Conv3d).get_fn()
		self.apply(fn)
		return


class LearnedSoftPlus(nn.Module):
	def __init__(
			self,
			beta: float = 1.0,
			threshold: float = 10.0,
	):
		super().__init__()
		self.log_beta = torch.nn.Parameter(
			torch.tensor(float(beta)).log(),
			requires_grad=True,
		)
		self.threshold = threshold

	def forward(self, x):
		beta = self.log_beta.exp()
		beta_x = beta * x
		return torch.where(
			condition=beta_x < self.threshold,
			input=torch.log1p(beta_x.exp()) / beta,
			other=x,
		)


def process_ftrs(
		ftr: dict,
		scale: int = 4,
		pool: str = 'max',
		act_fn: str = 'swish', ):
	# activation
	activation = get_act_fn(act_fn)
	ftr = {
		s: activation(x) for
		s, x in ftr.items()
	}
	# pool
	if pool == 'max':
		pool = nn.AdaptiveMaxPool2d(scale)
	elif pool == 'avg':
		pool = nn.AdaptiveAvgPool2d(scale)
	elif pool == 'none':
		pool = None
	else:
		raise NotImplementedError
	if pool is not None:
		for s, x in ftr.items():
			if s > scale:
				ftr[s] = pool(x)
	return ftr
