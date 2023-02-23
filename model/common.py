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
		dim: int = 1,
		b: int = None, ):
	if b is None:
		error = torch.norm(true-pred, p=2, dim=dim).sum()
	else:
		error = 0.0
		num = int(np.ceil(len(true) / b))
		for i in range(num):
			ids = range(i*b, min((i+1)*b, len(true)))
			delta = (true - pred)[ids]
			error += torch.norm(
				delta, p=2, dim=dim).sum()
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
		return Identity()
	elif stride in [2, 4]:
		return FactorizedReduce(ci, int(cmult*ci))
	elif stride == -1:
		return nn.Sequential(
			nn.Upsample(
				scale_factor=cmult,
				mode='bilinear',
				align_corners=True),
			nn.Conv2d(
				in_channels=ci,
				out_channels=int(ci/cmult),
				kernel_size=1),
		)
	else:
		raise NotImplementedError(stride)


def get_act_fn(fn: str, inplace: bool = True):
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


# noinspection PyMethodMayBeStatic
class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, x):
		return x


class FactorizedReduce(nn.Module):
	def __init__(self, ci: int, co: int, dim: int = 2):
		super(FactorizedReduce, self).__init__()
		n_ops = 2 ** dim
		assert co % 2 == 0 and co > n_ops
		kwargs = {
			'in_channels': ci,
			'out_channels': co//n_ops,
			'kernel_size': 3,
			'stride': co//ci,
			'padding': 1,
			'bias': True,
		}
		self.swish = nn.SiLU(inplace=True)
		self.ops = nn.ModuleList()
		for i in range(n_ops - 1):
			if dim == 2:
				self.ops.append(nn.Conv2d(**kwargs))
			else:
				raise NotImplementedError
		kwargs['out_channels'] = co - len(self.ops) * (co//n_ops)
		if dim == 2:
			self.ops.append(nn.Conv2d(**kwargs))
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


def add_wn(m: nn.Module):
	if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
		nn.utils.weight_norm(m)


def add_sn(m: nn.Module):
	if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
		nn.utils.spectral_norm(m)


def get_norm(norm: str):
	if norm == 'batch':
		return nn.BatchNorm3d
	elif norm == 'group':
		return nn.GroupNorm
	elif norm == 'layer':
		return nn.LayerNorm
	else:
		return None


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
