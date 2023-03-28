from .utils_model import *
MULT = 2


@torch.jit.script
def endpoint_error(
		true: torch.Tensor,
		pred: torch.Tensor,
		dim: int = 1, ):
	epe = torch.linalg.norm(
		true - pred, dim=dim)
	epe = torch.sum(epe, dim=[1, 2])
	return epe


def endpoint_error_batch(
		true: torch.Tensor,
		pred: torch.Tensor,
		batch: int = 512,
		dim: int = 1, ):
	delta = true - pred
	epe = []
	n = int(np.ceil(len(true) / batch))
	for i in range(n):
		a = i * batch
		b = min((i+1) * batch, len(true))
		epe.append(torch.linalg.norm(
			delta[range(a, b)], dim=dim,
		))
	epe = torch.cat(epe)
	epe = torch.sum(epe, dim=[1, 2])
	return epe


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
				mode='nearest'),
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
	def __init__(self, ci: int, co: int):
		super(FactorizedReduce, self).__init__()
		assert co % 2 == 0 and co > 4
		co_each = co // 4
		kwargs = {
			'in_channels': ci,
			'out_channels': co_each,
			'kernel_size': 1,
			'stride': co // ci,
			'padding': 0,
			'bias': True,
		}
		self.swish = nn.SiLU(inplace=True)
		self.ops = nn.ModuleList()
		for i in range(3):
			self.ops.append(Conv2D(**kwargs))
		kwargs['out_channels'] = co - 3 * co_each
		self.ops.append(Conv2D(**kwargs))

	def forward(self, x):
		x = self.swish(x)
		idx, out = 0, []
		for op in self.ops:
			i, j = idx // 2, idx % 2
			out.append(op(x[..., i:, j:]))
			idx += 1
		return torch.cat(out, dim=1)


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


class Cell(nn.Module):
	def __init__(
			self,
			ci: int,
			co: int,
			n_nodes: int,
			cell_type: str,
			act_fn: str,
			use_bn: bool,
			use_se: bool,
			scale: float = None,
			**kwargs,
	):
		super(Cell, self).__init__()
		self.skip = get_skip_connection(
			ci, MULT, cell_type)
		self.ops = nn.ModuleList()
		for i in range(n_nodes):
			op = ConvLayer(
				ci=ci if i == 0 else co,
				co=co,
				stride=get_stride(cell_type, MULT)
				if i == 0 else 1,
				act_fn=act_fn,
				use_bn=use_bn,
				init_scale=scale
				if i+1 == n_nodes
				else None,
				**kwargs,
			)
			self.ops.append(op)
		if use_se:
			self.se = SELayer(co, act_fn)
		else:
			self.se = None

	def forward(self, x):
		skip = self.skip(x)
		for op in self.ops:
			x = op(x)
		if self.se is not None:
			x = self.se(x)
		return skip + x


class ConvLayer(nn.Module):
	def __init__(
			self,
			ci: int,
			co: int,
			stride: int,
			act_fn: str,
			use_bn: bool,
			**kwargs,
	):
		super(ConvLayer, self).__init__()
		defaults = {
			'in_channels': ci,
			'out_channels': co,
			'kernel_size': 3,
			'normalize_dim': 0,
			'init_scale': None,
			'stride': abs(stride),
			'padding': 1,
			'dilation': 1,
			'groups': 1,
			'bias': True,
		}
		if stride == -1:
			self.upsample = nn.Upsample(
				scale_factor=MULT,
				mode='nearest',
			)
		else:
			self.upsample = None
		if use_bn:
			self.bn = nn.BatchNorm2d(ci)
		else:
			self.bn = None
		self.act_fn = get_act_fn(act_fn, False)
		kwargs = setup_kwargs(defaults, kwargs)
		self.conv = Conv2D(**kwargs)

	def forward(self, x):
		if self.bn is not None:
			x = self.bn(x)
		if self.act_fn is not None:
			x = self.act_fn(x)
		if self.upsample is not None:
			x = self.upsample(x)
		x = self.conv(x)
		return x


class Conv2D(nn.Conv2d):
	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			kernel_size: int,
			normalize_dim: int = 0,
			init_scale: float = None,
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
		init = torch.ones(self.out_channels)
		if init_scale is not None:
			init *= init_scale
		self.log_weight_norm = nn.Parameter(
			data=torch.log(init),
			requires_grad=True,
		)
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
			normalize_dim: int = 1,
			init_scale: float = None,
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
		init = torch.ones(self.out_channels)
		if init_scale is not None:
			init *= init_scale
		self.log_weight_norm = nn.Parameter(
			data=torch.log(init),
			requires_grad=True,
		)
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
			torch.log(init + 1e-2),
			requires_grad=True,
		)
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


class RotConv2d(nn.Conv2d):
	def __init__(
			self,
			co: int,
			n_rots: int,
			kernel_size: Union[int, Iterable[int]],
			bias: bool = True,
			gain: bool = True,
			**kwargs,
	):
		super(RotConv2d, self).__init__(
			in_channels=2,
			out_channels=co,
			kernel_size=kernel_size,
			padding='valid',
			bias=bias,
			**kwargs,
		)
		self.n_rots = n_rots
		self._build_rot_mat()
		if bias:
			bias = nn.Parameter(
				torch.zeros(co*n_rots),
				requires_grad=True,
			)
		else:
			bias = None
		self.bias = bias
		if gain:
			gain = nn.Parameter(
				torch.ones(co*n_rots),
				requires_grad=True,
			)
		else:
			gain = None
		self.gain = gain
		self.w = self.augment_weight()

	def forward(self, x):
		self.w = self.augment_weight()
		return F.conv2d(
			input=x,
			weight=self.w,
			bias=self.bias,
			stride=self.stride,
			padding=self.padding,
			dilation=self.dilation,
			groups=self.groups,
		)

	def _build_rot_mat(self):
		thetas = np.deg2rad(np.arange(
			0, 360, 360 / self.n_rots))
		u = [0.0, 0.0, 1.0]
		u = np.array(u).reshape(1, -1)
		u = np.repeat(u, self.n_rots, 0)
		u *= thetas.reshape(-1, 1)
		r = Rotation.from_rotvec(u)
		r = r.as_matrix()
		r = torch.tensor(
			data=r[:, :2, :2],
			dtype=torch.float,
		)
		self.register_buffer('rot_mat', r)
		return

	def augment_weight(self, eps=1e-12):
		wn = torch.linalg.vector_norm(
			x=self.weight,
			dim=[1, 2, 3],
			keepdim=True,
		)
		w = torch.einsum(
			'rij, kjxy -> krixy',
			self.rot_mat,
			self.weight / (wn + eps),
		).flatten(end_dim=1)
		if self.gain is not None:
			w *= self.gain.view(-1, 1, 1, 1)
		return w


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


def _dims(normalize_dim, ndims):
	assert normalize_dim in [0, 1]
	dims = list(range(ndims))
	shape = [
		1 if i != normalize_dim
		else -1 for i in dims
	]
	dims.pop(normalize_dim)
	return dims, shape


def _normalize(lognorm, weight, shape, dims, eps=1e-12):
	n = torch.exp(lognorm).view(shape)
	wn = torch.linalg.vector_norm(
		x=weight, dim=dims, keepdim=True)
	return n * weight / (wn + eps)
