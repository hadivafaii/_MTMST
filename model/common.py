from .utils_model import *


def gaussian_residual_kl(delta_mu, log_deltasigma, logsigma):
	"""
	:param delta_mu: residual mean
	:param log_deltasigma: log of residual covariance
	:param logsigma: log of prior covariance
	:return: D_KL ( q || p ) where
		q = N ( mu + delta_mu , sigma * deltasigma ), and
		p = N ( mu, sigma )
	"""
	return 0.5 * (
			delta_mu ** 2 / logsigma.exp()
			+ log_deltasigma.exp()
			- log_deltasigma - 1.0
	).sum()


def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
	# computes D_KL ( 1 || 2 )
	return 0.5 * (  # This is wrong, it should be (logsigma1 / logsigma2).exp()
			(logsigma1.exp() + (mu1 - mu2) ** 2) / logsigma2.exp()
			+ logsigma2 - logsigma1 - 1.0
	).sum()


def endpoint_error(
		true: torch.Tensor,
		pred: torch.Tensor,
		w: torch.Tensor = None,
		batch_size: int = None,
		dim: int = 1, ):
	delta = true - pred
	if batch_size is None:
		error = torch.linalg.norm(delta, dim=dim)
	else:
		error = []
		n = int(np.ceil(len(true) / batch_size))
		for i in range(n):
			a = i * batch_size
			b = min((i+1) * batch_size, len(true))
			error.append(torch.linalg.norm(
				delta[range(a, b)], dim=dim,
			))
		error = torch.cat(error)
	if w is not None:
		error = error.sum(-1).sum(-1)
		error = (error * w).sum() / w.mean()
	else:
		error = error.sum()
	return error


def get_stride(cell_type: str, cmult: int):
	startswith = cell_type.split('_')[0]
	if startswith in ['normal', 'combiner']:
		stride = 1
	elif startswith == 'down':
		stride = cmult
	elif startswith == 'up':
		stride = -1
	else:
		raise NotImplementedError(cell_type)
	return stride


def get_skip_connection(
		ci: int,
		cmult: int,
		stride: Union[int, str], ):
	if isinstance(stride, str):
		stride = get_stride(stride, cmult)
	if stride == 1:
		return nn.Identity()
	elif stride in [2, 4]:
		return FactorizedReduce(ci, int(cmult*ci))
	elif stride == -1:
		return nn.Sequential(
			nn.Upsample(
				scale_factor=cmult,
				mode='bilinear',
				align_corners=True),
			Conv2D(
				in_channels=ci,
				out_channels=int(ci/cmult),
				kernel_size=1),
		)
	else:
		raise NotImplementedError(stride)


def get_act_fn(fn: str, inplace: bool = False):
	if fn == 'none':
		return None
	elif fn == 'relu':
		return nn.ReLU(inplace=inplace)
	elif fn == 'swish':
		return nn.SiLU(inplace=inplace)
	elif fn == 'elu':
		return nn.ELU(inplace=inplace)
	else:
		raise NotImplementedError(fn)


class FactorizedReduce(nn.Module):
	def __init__(self, ci: int, co: int, dim: int = 2):
		super(FactorizedReduce, self).__init__()
		n_ops = 2 ** dim
		assert co % 2 == 0 and co > n_ops
		kwargs = {
			'in_channels': ci,
			'out_channels': co//n_ops,
			'kernel_size': 1,
			'stride': co//ci,
			'padding': 0,
			'bias': True,
		}
		self.swish = nn.SiLU(inplace=True)
		self.ops = nn.ModuleList()
		for i in range(n_ops - 1):
			if dim == 2:
				self.ops.append(Conv2D(**kwargs))
			else:
				raise NotImplementedError
		kwargs['out_channels'] = co - len(self.ops) * (co//n_ops)
		if dim == 2:
			self.ops.append(Conv2D(**kwargs))
		else:
			raise NotImplementedError

	def forward(self, x):
		x = self.swish(x)
		y = []
		for ii, op in enumerate(self.ops, start=1):
			i, j, k = base2(len(self.ops) - ii)
			y.append(op(x[..., i:, j:, k:]))
		return torch.cat(y, dim=1)


class SELayer(nn.Module):
	def __init__(
			self,
			ci: int,
			act_fn: str,
			reduc: int = 16, ):
		super(SELayer, self).__init__()
		self.hdim = max(ci // reduc, 4)
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
			nn.Linear(ci, self.hdim), get_act_fn(act_fn),
			nn.Linear(self.hdim, ci), nn.Sigmoid(),
		)

	def forward(self, x):
		b, c, _, _ = x.size()
		y = self.pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1, 1)
		return x * y.expand_as(x)


class Conv2D(nn.Conv2d):
	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			kernel_size: Union[int, Tuple[int, int]],
			normalize_dim: int = 0,
			**kwargs,
	):
		kwargs = filter_kwargs(nn.Conv2d, kwargs)
		super(Conv2D, self).__init__(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=kernel_size,
			**kwargs,
		)
		self.dims, self.shape = _dims(normalize_dim, 4)
		init = torch.linalg.vector_norm(
			x=self.weight, dim=self.dims)
		self.log_weight_norm = nn.Parameter(
			torch.log(init + 1e-2), requires_grad=True)
		self.w = self.normalize_weight()

	def forward(self, x):
		self.w = self.normalize_weight()
		return F.conv2d(
			input=x,
			weight=self.w,
			bias=self.bias,
			stride=self.stride,
			padding=self.padding,
			dilation=self.dilation,
			groups=self.groups,
		)

	def normalize_weight(self):
		return _normalize(
			lognorm=self.log_weight_norm,
			weight=self.weight,
			shape=self.shape,
			dims=self.dims,
		)


class DeConv2D(nn.ConvTranspose2d):
	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			kernel_size: Union[int, Tuple[int, int]],
			normalize_dim: int = 0,
			**kwargs,
	):
		kwargs = filter_kwargs(nn.ConvTranspose2d, kwargs)
		super(DeConv2D, self).__init__(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=kernel_size,
			**kwargs,
		)
		self.dims, self.shape = _dims(normalize_dim, 4)
		init = torch.linalg.vector_norm(
			x=self.weight, dim=self.dims)
		self.log_weight_norm = nn.Parameter(
			torch.log(init + 1e-2), requires_grad=True)
		self.w = self.normalize_weight()

	def forward(self, x, output_size=None):
		self.w = self.normalize_weight()
		return F.conv_transpose2d(
			input=x,
			weight=self.w,
			bias=self.bias,
			stride=self.stride,
			padding=self.padding,
			dilation=self.dilation,
			groups=self.groups,
		)

	def normalize_weight(self):
		return _normalize(
			lognorm=self.log_weight_norm,
			weight=self.weight,
			shape=self.shape,
			dims=self.dims,
		)


class Linear(nn.Linear):
	def __init__(
			self,
			in_features: int,
			out_features: int,
			normalize_dim: int = 0,
			**kwargs,
	):
		kwargs = filter_kwargs(nn.Linear, kwargs)
		super(Linear, self).__init__(
			in_features=in_features,
			out_features=out_features,
			**kwargs,
		)
		self.dims, self.shape = _dims(normalize_dim, 2)
		init = torch.linalg.vector_norm(
			x=self.weight, dim=self.dims)
		self.log_weight_norm = nn.Parameter(
			torch.log(init + 1e-2), requires_grad=True)
		self.w = self.normalize_weight()

	def forward(self, x):
		self.w = self.normalize_weight()
		return F.linear(
			input=x,
			weight=self.w,
			bias=self.bias,
		)

	def normalize_weight(self):
		return _normalize(
			lognorm=self.log_weight_norm,
			weight=self.weight,
			shape=self.shape,
			dims=self.dims,
		)


class AddNorm(object):
	def __init__(self, norm, types, **kwargs):
		super(AddNorm, self).__init__()
		self.norm = norm
		self.types = types
		if self.norm == 'spectral':
			self.kwargs = filter_kwargs(
				fn=nn.utils.parametrizations.spectral_norm,
				kw=kwargs,
			)
		elif self.norm == 'weight':
			self.kwargs = filter_kwargs(
				fn=nn.utils.weight_norm,
				kw=kwargs,
			)
		else:
			raise NotImplementedError

	def get_fn(self) -> Callable:
		if self.norm == 'spectral':
			def fn(m):
				if isinstance(m, self.types):
					nn.utils.parametrizations.spectral_norm(
						module=m, **self.kwargs)
				return
		elif self.norm == 'weight':
			def fn(m):
				if isinstance(m, self.types):
					nn.utils.weight_norm(
						module=m, **self.kwargs)
				return
		else:
			raise NotImplementedError
		return fn


def get_init_fn(init_range: float = 0.01):
	def init_weights(m: nn.Module):
		if 'Conv2d' in m.__class__.__name__:
			nn.init.kaiming_normal_(
				tensor=m.weight,
				mode='fan_out',
				nonlinearity='relu',
			)
		elif 'Norm' in m.__class__.__name__:
			nn.init.constant_(
				tensor=m.weight,
				val=1.0,
			)
		elif isinstance(m, nn.Linear):
			nn.init.normal_(
				tensor=m.weight,
				mean=0.0,
				std=init_range,
			)
		else:
			pass
		if hasattr(m, 'bias') and m.bias is not None:
			nn.init.constant_(
				tensor=m.bias,
				val=0.0,
			)
	return init_weights


def _dims(normalize_dim, ndims):
	assert normalize_dim in [0, 1]
	dims = list(range(ndims))
	shape = [
		1 if i != normalize_dim
		else -1 for i in dims
	]
	dims.pop(normalize_dim)
	return dims, shape


def _normalize(lognorm, weight, shape, dims, eps=1e-6):
	n = torch.exp(lognorm).view(shape)
	wn = torch.linalg.vector_norm(
		x=weight, dim=dims, keepdim=True)
	return n * weight / (wn + eps)
