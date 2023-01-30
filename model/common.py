from utils.model import *


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
	return 0.5 * (
			(logsigma1.exp() + (mu1 - mu2) ** 2) / logsigma2.exp()
			+ logsigma2 - logsigma1 - 1.0
	).sum()


def compute_endpoint_error(
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


def to_np(x: torch.Tensor):
	if isinstance(x, np.ndarray):
		return x
	return x.data.cpu().numpy()


def reparametrize(mu, logsigma):
	std = torch.exp(0.5 * logsigma)
	eps = torch.randn_like(mu).to(mu.device)
	z = mu + std * eps
	return z


def conv3x3x3(
		ci,
		co,
		stride=1,
		padding=1,
		dilation=1,
		groups=1,
		bias=True, ):
	return nn.Conv3d(
		in_channels=ci,
		out_channels=co,
		kernel_size=3,
		stride=stride,
		padding=padding,
		dilation=dilation,
		groups=groups,
		bias=bias,
	)


def conv1x1x1(
		ci,
		co,
		stride=1,
		bias=True, ):
	return nn.Conv3d(
		in_channels=ci,
		out_channels=co,
		kernel_size=1,
		stride=stride,
		padding=0,
		bias=bias,
	)


def deconv3x3x3(ci, co, stride=1, groups=1, dilation=1, bias=True):
	return nn.ConvTranspose3d(
		in_channels=ci,
		out_channels=co,
		kernel_size=3,
		stride=stride,
		padding=dilation,
		dilation=dilation,
		groups=groups,
		bias=bias,
	)


def deconv1x1x1(ci, co, stride=1, bias=False):
	return nn.ConvTranspose3d(
		in_channels=ci,
		out_channels=co,
		kernel_size=1,
		stride=stride,
		bias=bias,
	)


def get_skip_connection(
		ci: int,
		cmult: int,
		stride: int, ):
	if stride == 1:
		return Identity()
	elif stride == 2:
		return FactorizedReduce(ci, int(cmult*ci))
	elif stride == -1:
		return nn.Sequential(
			nn.Upsample(scale_factor=2),
			conv1x1x1(ci, int(ci/cmult)),
		)
	else:
		raise NotImplementedError(stride)


# noinspection PyMethodMayBeStatic
class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, x):
		return x


class FactorizedReduce(nn.Module):
	def __init__(self, ci: int, co: int, dim: int = 3):
		super(FactorizedReduce, self).__init__()
		n_ops = 2 ** dim
		assert co % 2 == 0 and co > n_ops
		kwargs = {
			'ci': ci,
			'co': co//n_ops,
			'stride': 2,
			'padding': 1,
			'bias': True,
		}
		self.swish = nn.SiLU(inplace=True)
		self.ops = nn.ModuleList()
		for i in range(n_ops - 1):
			if dim == 3:
				self.ops.append(conv3x3x3(**kwargs))
			else:
				raise NotImplementedError
		kwargs['co'] = co - len(self.ops) * (co//n_ops)
		if dim == 3:
			self.ops.append(conv3x3x3(**kwargs))
		else:
			raise NotImplementedError

	def forward(self, x):
		x = self.swish(x)
		y = []
		for ii, op in enumerate(self.ops, start=1):
			i, j, k = base2(len(self.ops) - ii)
			y.append(op(x[..., i:, j:, k:]))
		return torch.cat(y, dim=1)


# TODO: this doesnt work for 3D volumetric data
class UpSample(nn.Module):
	def __init__(self, **kwargs):
		super(UpSample, self).__init__()
		defaults = {
			'scale_factor': 2,
			'mode': 'bilinear',
			'align_corners': True,
			'antialias': False,
		}
		self.kwargs = setup_kwargs(defaults, kwargs)

	def forward(self, x):
		return F.interpolate(x, **self.kwargs)


class SELayer(nn.Module):
	def __init__(self, ci: int, reduc: int = 16):
		super(SELayer, self).__init__()
		self.hdim = max(ci // reduc, 4)
		self.avg_pool = nn.AdaptiveAvgPool3d(1)
		self.fc = nn.Sequential(
			nn.Linear(ci, self.hdim), nn.SiLU(True),
			nn.Linear(self.hdim, ci), nn.Sigmoid(),
		)

	def forward(self, x):
		b, c, _, _, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1, 1, 1)
		return x * y.expand_as(x)


class Module(nn.Module):
	def __init__(self, cfg, verbose: bool = False):
		super(Module, self).__init__()
		self.cfg = cfg
		self.datetime = now(True)
		self.verbose = verbose

	def print(self):
		print_num_params(self)


def add_wn(m: nn.Module):
	if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
		nn.utils.weight_norm(m)


def add_sn(m: nn.Module):
	if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
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
		if 'Conv3d' in m.__class__.__name__:
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
