from .common import *


_MULT = 2


class VAE(Module):
	def __init__(self, cfg: ConfigVAE, **kwargs):
		super(VAE, self).__init__(cfg, **kwargs)

		self.beta = 1.0
		self._init()

		self.apply(get_init_fn(cfg.init_range))
		if cfg.conv_norm == 'weight':
			self.apply(add_wn)
		elif cfg.conv_norm == 'spectral':
			self.apply(add_sn)
		if self.verbose:
			self.print()

	def forward(self, x):
		raise NotImplementedError

	def _init(self):
		self._init_stem()
		self._init_sizes()
		mult = self._init_pre(1)
		mult = self._init_enc(mult)
		mult = self._init_enc0(mult)
		mult = self._init_dec(mult)
		return

	def _init_sizes(self):
		self.n_ch = self.stem.out_channels * self.stem.n_rots
		c_scaling = _MULT ** (self.cfg.n_pre_blocks + self.cfg.n_latent_scales - 1)
		s_scaling = 2 ** (self.cfg.n_pre_blocks + self.cfg.n_latent_scales - 1)
		self.z0_sz = [
			self.cfg.n_latent_per_group,
			self.cfg.input_sz // s_scaling,
			self.cfg.input_sz // s_scaling,
		]
		prior_ftr0_sz = (
			c_scaling * self.n_ch,
			self.cfg.input_sz // s_scaling,
			self.cfg.input_sz // s_scaling,
		)
		self.prior_ftr0 = nn.Parameter(
			data=torch.rand(prior_ftr0_sz),
			requires_grad=True,
		)
		return

	def _init_stem(self):
		self.stem = RotConv3d(
			co=self.cfg.n_kers,
			n_rots=self.cfg.n_rots,
			kernel_size=self.cfg.ker_sz,
			padding='same',
			bias=True,
		)
		return

	def _init_pre(self, mult):
		pre = nn.ModuleList()
		for _ in range(self.cfg.n_pre_blocks):
			for c in range(self.cfg.n_pre_cells):
				ch = int(self.n_ch * mult)
				if c == self.cfg.n_pre_cells - 1:
					co = _MULT * ch
					cell = Cell(ch, co, 2, 'down_pre', self.cfg.use_se)
					mult *= _MULT
				else:
					cell = Cell(ch, ch, 2, 'normal_pre', self.cfg.use_se)
				pre.append(cell)
		self.pre = pre
		return mult

	def _init_enc(self, mult):
		enc = nn.ModuleList()
		for s in range(self.cfg.n_latent_scales):
			ch = int(self.n_ch * mult)
			for g in range(self.cfg.groups[s]):
				for _ in range(self.cfg.n_cells_per_cond):
					enc.append(Cell(
						ch, ch, 2, 'normal_pre', self.cfg.use_se))
				# add encoder combiner
				combiner = not (
						g == self.cfg.groups[s] - 1 and
						s == self.cfg.n_latent_scales - 1
				)
				if combiner:
					enc.append(EncCombiner(ch, ch))

			# down cells after finishing a scale
			if s < self.cfg.n_latent_scales - 1:
				cell = Cell(ch, ch * _MULT, 2, 'down_enc', self.cfg.use_se)
				enc.append(cell)
				mult *= _MULT
		self.enc = enc
		return mult

	def _init_enc0(self, mult):
		ch = int(self.n_ch * mult)
		self.enc0 = nn.Sequential(
			nn.ELU(inplace=True),
			conv1x1x1(ch, ch, bias=True),
			nn.ELU(inplace=True),
		)
		return mult

	def _init_sampler(self, mult):
		enc_sampler = nn.ModuleList()
		dec_sampler = nn.ModuleList()
		for s in range(self.n_latent_scales):
			ch = int(self.n_ch * mult)
			for g in range(self.groups[self.n_latent_scales - s - 1]):
				# build mu, sigma generator for encoder
				enc_sampler.append(conv3x3x3(
					ch, 2 * self.cfg.n_latent_per_group))
				# for the first group, we use a fixed standard Normal.
				if not (s == 0 and g == 0):
					dec_sampler.append(nn.Sequential(
						nn.ELU(inplace=True),
						conv1x1x1(ch, 2 * self.cfg.n_latent_per_group),
					))
			mult /= _MULT
		self.enc_sampler = enc_sampler
		self.dec_sampler = dec_sampler
		# TODO: delete these lines
		# self.enc_kv = nn.ModuleList()
		# self.dec_kv = nn.ModuleList()
		# self.query = nn.ModuleList()
		return mult

	def _init_dec(self, mult):
		dec = nn.ModuleList()
		for s in range(self.cfg.n_latent_scales):
			ch = int(self.n_ch * mult)
			for g in range(self.cfg.groups[self.cfg.n_latent_scales - s - 1]):
				if not (s == 0 and g == 0):
					for _ in range(self.cfg.n_cells_per_cond):
						dec.append(Cell(
							ch, ch, 1, 'normal_dec', self.cfg.use_se))
				dec.append(DecCombiner(
					ch, self.cfg.n_latent_per_group, ch))

			# down cells after finishing a scale
			if s < self.cfg.n_latent_scales - 1:
				dec.append(Cell(
					ch, int(ch / _MULT), 1, 'up_dec', self.cfg.use_se))
				mult /= _MULT
		self.dec = dec
		return mult


class Cell(nn.Module):
	def __init__(
			self,
			ci: int,
			co: int,
			n_nodes: int,
			cell_type: str,
			use_se: bool, ):
		super(Cell, self).__init__()

		stride = get_stride_for_cell_type(cell_type)
		self.skip = get_skip_connection(ci, _MULT, stride)

		self.ops = nn.ModuleList()
		for i in range(n_nodes):
			stride = get_stride_for_cell_type(cell_type) if i == 0 else 1
			op = BNSwishConv(ci if i == 0 else co, co, stride)
			self.ops.append(op)
		if use_se:
			self.se = SELayer(co)
		else:
			self.se = None

	def forward(self, x):
		skip = self.skip(x)
		for i, op in enumerate(self.ops):
			x = op(x)
		if self.se is not None:
			x = self.se(x)
		return skip + 0.1 * x


class EncCombiner(nn.Module):
	def __init__(self, ci: int, co: int):
		super(EncCombiner, self).__init__()
		self.conv = conv1x1x1(ci, co)

	def forward(self, x1, x2):
		return x1 + self.conv(x2)


class DecCombiner(nn.Module):
	def __init__(self, ci1, ci2, co):
		super(DecCombiner, self).__init__()
		self.conv = conv1x1x1(ci1 + ci2, co)

	def forward(self, x1, x2):
		x = torch.cat([x1, x2], dim=1)
		return self.conv(x)


class BNSwishConv(nn.Module):
	def __init__(
			self,
			ci: int,
			co: int,
			stride: int,
			**kwargs, ):
		super(BNSwishConv, self).__init__()
		self.upsample = stride == -1
		defaults = {
			'in_channels': ci,
			'out_channels': co,
			'kernel_size': 3,
			'stride': abs(stride),
			'padding': 1,
			'dilation': 1,
			'groups': 1,
			'bias': True,
		}
		self.bn = nn.BatchNorm3d(ci)
		self.swish = nn.SiLU(inplace=True)
		self.conv = nn.Conv3d(**setup_kwargs(defaults, kwargs))

	def forward(self, x):
		x = self.bn(x)
		x = self.swish(x)
		if self.upsample:
			x = F.interpolate(
				input=x,
				scale_factor=2,
				mode='nearest',
			)
		x = self.conv(x)
		return x


class ConvBNSwish(nn.Module):
	def __init__(
			self,
			ci: int,
			co: int,
			stride: int,
			**kwargs, ):
		super(ConvBNSwish, self).__init__()
		self.upsample = stride == -1
		defaults = {
			'in_channels': ci,
			'out_channels': co,
			'kernel_size': 3,
			'stride': abs(stride),
			'padding': 1,
			'dilation': 1,
			'groups': 1,
			'bias': True,
		}
		self.bn = nn.BatchNorm3d(ci)
		self.swish = nn.SiLU(inplace=True)
		self.conv = nn.Conv3d(**setup_kwargs(defaults, kwargs))

	def forward(self, x):
		x = self.bn(x)
		x = self.swish(x)
		if self.upsample:
			x = F.interpolate(
				input=x,
				scale_factor=2,
				mode='nearest',
			)
		x = self.conv(x)
		return x


# TODO: InvertedResidual


class RotConv3d(nn.Conv3d):
	def __init__(
			self,
			co: int,
			n_rots: int,
			kernel_size: Union[int, Iterable[int]],
			stride: Union[int, Iterable[int]] = 1,
			padding: Union[int, Iterable[int], str] = 'same',
			dilation: Union[int, Iterable[int]] = 1,
			groups: int = 1,
			bias: bool = True,
	):
		super(RotConv3d, self).__init__(
			in_channels=2,
			out_channels=co,
			kernel_size=kernel_size,
			stride=stride,
			padding=padding,
			dilation=dilation,
			groups=groups,
			bias=bias,
		)
		self.n_rots = n_rots
		self._build_rot_mat()
		if bias:
			self.bias = nn.Parameter(
				torch.zeros(co * n_rots))

	def forward(self, x):
		return F.conv3d(
			input=x,
			weight=self._get_augmented_weight(),
			bias=self.bias,
			stride=self.stride,
			padding=self.padding,
			dilation=self.dilation,
			groups=self.groups,
		)

	def _build_rot_mat(self):
		thetas = np.deg2rad(np.arange(0, 360, 360/self.n_rots))
		c, s = np.cos(thetas), np.sin(thetas)
		rot_mat = np.array([[c, -s], [s, c]])
		rot_mat = torch.tensor(
			data=rot_mat,
			dtype=torch.float,
		).permute(2, 0, 1)
		self.register_buffer('rot_mat', rot_mat)
		return

	def _get_augmented_weight(self):
		w = torch.einsum(
			'jkn, inlmo -> ijklmo',
			self.rot_mat,
			self.weight,
		)
		return w.flatten(end_dim=1)
