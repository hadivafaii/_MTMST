from .common import *
from .distributions import Normal

_MULT = 2


class VAE(Module):
	def __init__(self, cfg: ConfigVAE, **kwargs):
		super(VAE, self).__init__(cfg, **kwargs)

		self.beta = 1.0
		self._init()

		# self.apply(get_init_fn(cfg.init_range))
		if cfg.conv_norm == 'weight':
			self.apply(add_wn)
		elif cfg.conv_norm == 'spectral':
			self.apply(add_sn)
		if self.verbose:
			self.print()

	def forward(self, x):
		s = self.stem(x)
		print(f'after stem, s: {s.size()}')

		# perform pre-processing
		for cell in self.pre:
			s = cell(s)
		print(f'after pre, s: {s.size()}')

		# run the main encoder tower
		combiners_enc = []
		combiners_s = []
		for cell in self.enc_tower:
			if isinstance(cell, EncCombiner):
				combiners_enc.append(cell)
				combiners_s.append(s)
			else:
				s = cell(s)
				print(s.size())

		# reverse combiner cells and their input for decoder
		combiners_enc.reverse()
		combiners_s.reverse()

		idx = 0
		ftr = self.enc0(s)  # this reduces the channel dimension
		param0 = self.enc_sampler[idx](ftr)
		mu_q, logsig_q = torch.chunk(param0, 2, dim=1)
		dist = Normal(mu_q, logsig_q)  # for the first approx. posterior
		z, _ = dist.sample()
		log_q_conv = dist.log_p(z)
		all_log_q = [log_q_conv]
		all_q = [dist]

		print('top')
		print(f"idx = {idx}, z size: {z.size()}")

		# prior for z0
		dist = Normal(
			mu=torch.zeros_like(z),
			logsigma=torch.zeros_like(z),
		)
		log_p_conv = dist.log_p(z)
		all_log_p = [log_p_conv]
		all_p = [dist]

		s = self.prior_ftr0.unsqueeze(0)
		s = s.expand(z.size(0), -1, -1, -1)

		print(list(s.size()))
		# for ll, aaa in enumerate(combiners_s):
		# print(f'hi, {ll}, {aaa.size()}')

		print('begin dec')
		for cell in self.dec_tower:
			if isinstance(cell, DecCombiner):
				if idx > 0:
					# form prior
					param = self.dec_sampler[idx - 1](s)
					mu_p, logsig_p = torch.chunk(param, 2, dim=1)

					# form encoder
					# print(f"ftr   ||   idx = {idx},   enc-s: {combiners_s[idx - 1].size()},   s: {s.size()}")
					ftr = combiners_enc[idx - 1](combiners_s[idx - 1], s)
					param = self.enc_sampler[idx](ftr)
					mu_q, logsig_q = torch.chunk(param, 2, dim=1)
					if self.cfg.residual_kl:
						dist = Normal(mu_p + mu_q, logsig_p + logsig_q)
					else:
						dist = Normal(mu_q, logsig_q)
					z, _ = dist.sample()
					log_q_conv = dist.log_p(z)
					all_log_q.append(log_q_conv)
					all_q.append(dist)

					# evaluate log_p(z)
					dist = Normal(mu_p, logsig_p)
					log_p_conv = dist.log_p(z)
					all_log_p.append(log_p_conv)
					all_p.append(dist)

				# 'combiner_dec'
				# print(f"DecCombiner   ||   idx = {idx},   s: {s.size()},   z: {z.size()}")
				s = cell(s, z)
				idx += 1
			else:
				s = cell(s)

		if self.vanilla:
			print('vailla')
			s = self.stem_decoder(z)

		# print('post')
		for cell in self.post:
			s = cell(s)
			# print(s.size())

		# compute kl
		log_p, log_q = 0., 0.
		kl_all, kl_diag = [], []
		for q, p, log_q_conv, log_p_conv in zip(all_q, all_p, all_log_q, all_log_p):
			kl_per_var = q.kl(p)
			kl_diag.append(torch.mean(torch.sum(kl_per_var, dim=[2, 3]), dim=0))
			kl_all.append(torch.sum(kl_per_var, dim=[1, 2, 3]))
			log_q += torch.sum(log_q_conv, dim=[1, 2, 3])
			log_p += torch.sum(log_p_conv, dim=[1, 2, 3])

		return self.out(s), log_p, log_q, kl_all, kl_diag

	def sample(self, n: int, t: float = 1.0, device: torch.device = None):
		z0_sz = [n] + self.z0_sz
		mu = torch.zeros(z0_sz)
		logsigma = torch.zeros(z0_sz)
		if device is not None:
			mu = mu.to(device)
			logsigma = logsigma.to(device)
		dist = Normal(mu, logsigma, temp=t)
		z, _ = dist.sample()

		idx = 0
		s = self.prior_ftr0.unsqueeze(0)
		s = s.expand(z.size(0), -1, -1, -1)
		for cell in self.dec_tower:
			if isinstance(cell, DecCombiner):
				if idx > 0:
					# form prior
					param = self.dec_sampler[idx - 1](s)
					mu, logsigma = torch.chunk(param, 2, dim=1)
					dist = Normal(mu, logsigma, t)
					z, _ = dist.sample()

				# 'combiner_dec'
				s = cell(s, z)
				idx += 1
			else:
				s = cell(s)

		if self.vanilla:
			s = self.stem_decoder(z)

		for cell in self.post:
			s = cell(s)

		return self.out(s)

	def _init(self):
		self.vanilla = (
				self.cfg.n_latent_scales == 1 and
				self.cfg.n_groups_per_scale == 1
		)
		self.kws = dict(
			act_fn=self.cfg.activation_fn,
			use_bn=self.cfg.use_bn,
			use_se=self.cfg.use_se,
		)
		self._init_stem()
		self._init_sizes()
		mult = self._init_pre(1)
		# print(mult, 'after _init_pre()')
		if not self.vanilla:
			mult = self._init_enc(mult)
		else:
			self.enc_tower = []
		# print(mult, 'after _init_enc()')
		mult = self._init_enc0(mult)
		# print(mult, 'after _init_enc0()')
		self._init_sampler(mult)
		# print(mult, 'after _init_sampler()')
		if not self.vanilla:
			mult = self._init_dec(mult)
			self.stem_decoder = None
		else:
			self.dec_tower = []
			self.stem_decoder = nn.Conv2d(
				in_channels=self.cfg.n_latent_per_group,
				out_channels=int(mult * self.n_ch),
				kernel_size=1,
			)
		# print(mult, 'after _init_dec()')
		mult = self._init_post(mult)
		# print(mult, 'after _init_post()')
		self._init_output(mult)
		return

	def _init_sizes(self):
		input_sz = self.cfg.input_sz - self.cfg.ker_sz + 1
		self.n_ch = self.stem.out_channels * self.stem.n_rots
		scaling = _MULT ** (
				self.cfg.n_pre_blocks +
				self.cfg.n_latent_scales - 1
		)
		self.z0_sz = [
			self.cfg.n_latent_per_group,
			input_sz // scaling,
			input_sz // scaling,
		]
		prior_ftr0_sz = [
			scaling * self.n_ch,
			input_sz // scaling,
			input_sz // scaling,
		]
		self.prior_ftr0 = nn.Parameter(
			data=torch.rand(prior_ftr0_sz),
			requires_grad=True,
		)
		return

	def _init_stem(self):
		self.stem = RotConv2d(
			co=self.cfg.n_kers,
			n_rots=self.cfg.n_rots,
			kernel_size=self.cfg.ker_sz,
			bias=True,
		)
		return

	def _init_pre(self, mult):
		pre = nn.ModuleList()
		looper = itertools.product(
			range(self.cfg.n_pre_blocks),
			range(self.cfg.n_pre_cells),
		)
		for _, c in looper:
			ch = int(self.n_ch * mult)
			if c == self.cfg.n_pre_cells - 1:
				co = _MULT * ch
				cell = Cell(
					ci=ch,
					co=co,
					n_nodes=2,
					cell_type='down_pre',
					**self.kws,
				)
				mult *= _MULT
			else:
				cell = Cell(
					ci=ch,
					co=ch,
					n_nodes=2,
					cell_type='normal_pre',
					**self.kws,
				)
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
						ci=ch,
						co=ch,
						n_nodes=2,
						cell_type='normal_pre',
						**self.kws,
					))
				# add encoder combiner
				combiner = not (
						g == self.cfg.groups[s] - 1 and
						s == self.cfg.n_latent_scales - 1
				)
				if combiner:
					enc.append(EncCombiner(ch, ch))

			# down cells after finishing a scale
			if s < self.cfg.n_latent_scales - 1:
				cell = Cell(
					ci=ch,
					co=ch * _MULT,
					n_nodes=2,
					cell_type='down_enc',
					**self.kws,
				)
				enc.append(cell)
				mult *= _MULT
		self.enc_tower = enc
		return mult

	def _init_enc0(self, mult):
		ch = int(self.n_ch * mult)
		self.enc0 = nn.Sequential(
			nn.ELU(inplace=True),
			nn.Conv2d(
				in_channels=ch,
				out_channels=ch,
				kernel_size=1,
				padding=0),
			nn.ELU(inplace=True),
		)
		return mult

	def _init_sampler(self, mult):
		kws = dict(
			out_channels=2 * self.cfg.n_latent_per_group,
			stride=1,
		)
		enc_sampler, dec_sampler = nn.ModuleList(), nn.ModuleList()
		for s in range(self.cfg.n_latent_scales):
			kws['in_channels'] = int(self.n_ch * mult)
			for g in range(self.cfg.groups[self.cfg.n_latent_scales - s - 1]):
				# build mu, sigma generator for encoder
				# TODO: add the if else stuff for when z size smaller than 3
				kws['kernel_size'] = 3
				kws['padding'] = 1
				enc_sampler.append(nn.Conv2d(**kws))
				# for 1st group we used a fixed standard Normal
				if not (s == 0 and g == 0):
					kws['kernel_size'] = 1
					kws['padding'] = 0
					dec_sampler.append(nn.Sequential(
						nn.ELU(inplace=True),
						nn.Conv2d(**kws),
					))
			mult /= _MULT
		self.enc_sampler, self.dec_sampler = enc_sampler, dec_sampler
		return

	def _init_dec(self, mult):
		dec = nn.ModuleList()
		for s in range(self.cfg.n_latent_scales):
			ch = int(self.n_ch * mult)
			for g in range(self.cfg.groups[self.cfg.n_latent_scales - s - 1]):
				if not (s == 0 and g == 0):
					for _ in range(self.cfg.n_cells_per_cond):
						dec.append(Cell(
							ci=ch,
							co=ch,
							n_nodes=1,
							cell_type='normal_dec',
							**self.kws,
						))
				dec.append(DecCombiner(
					ch, self.cfg.n_latent_per_group, ch))

			# down cells after finishing a scale
			if s < self.cfg.n_latent_scales - 1:
				dec.append(Cell(
					ci=ch,
					co=int(ch / _MULT),
					n_nodes=1,
					cell_type='up_dec',
					**self.kws,
				))
				mult /= _MULT
		self.dec_tower = dec
		return mult

	def _init_post(self, mult):
		post = nn.ModuleList()
		looper = itertools.product(
			range(self.cfg.n_post_blocks),
			range(self.cfg.n_post_cells),
		)
		for _, c in looper:
			ch = int(self.n_ch * mult)
			if c == 0:
				co = int(ch / _MULT)
				cell = Cell(
					ci=ch,
					co=co,
					n_nodes=1,
					cell_type='up_post',
					**self.kws,
				)
				mult /= _MULT
			else:
				cell = Cell(
					ci=ch,
					co=ch,
					n_nodes=1,
					cell_type='normal_post',
					**self.kws,
				)
			post.append(cell)
		self.post = post
		return mult

	def _init_output(self, mult):
		self.out = nn.Sequential(
			Conv(
				ci=int(self.n_ch * mult),
				co=2,
				stride=1,
				act_fn='elu',
				use_bn=self.kws['use_bn']),
			nn.ConvTranspose2d(
				in_channels=2,
				out_channels=2,
				kernel_size=self.cfg.ker_sz),
		)
		# print('inside _init_output()', int(self.n_ch * mult), mult)
		return


class EncCombiner(nn.Module):
	def __init__(self, ci: int, co: int):
		super(EncCombiner, self).__init__()
		self.conv = nn.Conv2d(
			in_channels=ci,
			out_channels=co,
			kernel_size=1,
		)

	def forward(self, x1, x2):
		return x1 + self.conv(x2)


class DecCombiner(nn.Module):
	def __init__(self, ci1, ci2, co):
		super(DecCombiner, self).__init__()
		self.conv = nn.Conv2d(
			in_channels=ci1+ci2,
			out_channels=co,
			kernel_size=1,
		)

	def forward(self, x1, x2):
		x = torch.cat([x1, x2], dim=1)
		return self.conv(x)


class Cell(nn.Module):
	def __init__(
			self,
			ci: int,
			co: int,
			n_nodes: int,
			cell_type: str,
			act_fn: str,
			use_bn: bool,
			use_se: bool, ):
		super(Cell, self).__init__()

		self.skip = get_skip_connection(ci, _MULT, cell_type)
		self.ops = nn.ModuleList()
		for i in range(n_nodes):
			op = Conv(
				ci=ci if i == 0 else co,
				co=co,
				stride=get_stride(cell_type, _MULT)
				if i == 0 else 1,
				act_fn=act_fn,
				use_bn=use_bn,
			)
			self.ops.append(op)
		if use_se:
			self.se = SELayer(co, act_fn)
		else:
			self.se = None

	def forward(self, x):
		skip = self.skip(x)
		for i, op in enumerate(self.ops):
			x = op(x)
		if self.se is not None:
			x = self.se(x)
		return skip + 0.1 * x


class Conv(nn.Module):
	def __init__(
			self,
			ci: int,
			co: int,
			stride: int,
			act_fn: str,
			use_bn: bool,
			**kwargs, ):
		super(Conv, self).__init__()
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
		if stride == -1:
			self.upsample = nn.Upsample(
				scale_factor=_MULT,
				mode='bilinear',
				align_corners=True,
			)
		else:
			self.upsample = None
		if use_bn:
			self.bn = nn.BatchNorm2d(ci)
		else:
			self.bn = None
		self.act_fn = get_act_fn(act_fn)
		kwargs = setup_kwargs(defaults, kwargs)
		kwargs = filter_kwargs(nn.Conv2d, kwargs)
		self.conv = nn.Conv2d(**kwargs)

	def forward(self, x):
		if self.bn is not None:
			x = self.bn(x)
		if self.act_fn is not None:
			x = self.act_fn(x)
		if self.upsample is not None:
			x = self.upsample(x)
		x = self.conv(x)
		return x


class RotConv2d(nn.Conv2d):
	def __init__(
			self,
			co: int,
			n_rots: int,
			kernel_size: Union[int, Iterable[int]],
			bias: bool = True,
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
				torch.zeros(co * n_rots))
		else:
			bias = None
		self.bias = bias

	def forward(self, x):
		return F.conv2d(
			input=x,
			weight=self._get_augmented_weight(),
			bias=self.bias,
			stride=self.stride,
			padding=self.padding,
			dilation=self.dilation,
			groups=self.groups,
		)

	def _build_rot_mat(self):
		thetas = np.deg2rad(np.arange(
			0, 360, 360 / self.n_rots))
		u = np.array([0.0, 0.0, 1.0]).reshape(1, -1)
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

	def _get_augmented_weight(self):
		w = torch.einsum(
			'rij, kjxy -> krixy',
			self.rot_mat,
			self.weight,
		)
		return w.flatten(end_dim=1)
