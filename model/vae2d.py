from .common import *
from .distributions import Normal

_MULT = 2


class VAE(Module):
	def __init__(self, cfg: ConfigVAE, **kwargs):
		super(VAE, self).__init__(cfg, **kwargs)
		self._init()
		if self.verbose:
			self.print()

	def forward(self, x):
		s = self.stem(x)
		# print(f'after stem, s: {s.size()}')

		# perform pre-processing
		for cell in self.pre:
			s = cell(s)

		# print(f'after pre, s: {s.size()}')

		# run the main encoder tower
		combiners_enc = []
		combiners_s = []
		for cell in self.enc_tower:
			if isinstance(cell, CombinerEnc):
				combiners_enc.append(cell)
				combiners_s.append(s)
			else:
				s = cell(s)
			# print('after', type(cell).__name__, s.size(), torch.isnan(s).sum())
			# print(s.size())

		# reverse combiner cells and their input for decoder
		combiners_enc.reverse()
		combiners_s.reverse()

		# print(s.size(), torch.isnan(s).sum())

		idx = 0
		ftr = self.enc0(s)  # this reduces the channel dimension
		# print('hi', idx, ftr.shape)
		# print(ftr)
		param0 = self.enc_sampler[idx](ftr)
		# print('hi', idx, param0.shape)
		# print(param0)
		mu_q, logsig_q = torch.chunk(param0, 2, dim=1)
		dist = Normal(mu_q, logsig_q)  # for the first approx. posterior
		z, _ = dist.sample()
		# print(z)
		# log_q_conv = dist.log_p(z)
		all_log_q = [dist.log_p(z)]  # was: [log_q_conv]
		all_q = [dist]
		latents = [z]

		# print('top')
		# print(f"idx = {idx}, z size: {z.size()}")

		# prior for z0
		dist = Normal(
			mu=torch.zeros_like(z),
			logsigma=torch.zeros_like(z),
		)
		log_p_conv = dist.log_p(z)
		all_log_p = [log_p_conv]
		all_p = [dist]

		# begin decoder pathway
		s = self.prior_ftr0.unsqueeze(0)
		s = s.expand(z.size(0), -1, -1, -1)
		for cell in self.dec_tower:
			if isinstance(cell, CombinerDec):
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
					latents.append(z)
					# if self.cfg.compress:
					# e = (-1,) * 2 + (s.size(-1),) * 2
					# z = z.expand(e)
					# log_q_conv = dist.log_p(z)
					all_log_q.append(dist.log_p(z))
					all_q.append(dist)

					# evaluate log_p(z)
					dist = Normal(mu_p, logsig_p)
					log_p_conv = dist.log_p(z)
					all_log_p.append(log_p_conv)
					all_p.append(dist)

				# 'combiner_dec'
				# print(f"CombinerDec   ||   idx = {idx},   s: {s.size()},   z: {z.size()}")
				s = cell(s, self.expand[idx](z))
				idx += 1
			else:
				s = cell(s)

		if self.vanilla:
			if self.verbose:
				print('vailla')
			s = self.stem_decoder(z)

		for cell in self.post:
			s = cell(s)

		# compute kl
		log_q, log_p, kl_all, kl_diag = self.loss_kl(
			all_q, all_p, all_log_q, all_log_p)
		return self.out(s), latents, log_q, log_p, kl_all, kl_diag

	def sample(self, n: int, t: float = 1.0, device: torch.device = None):
		z0_sz = [n] + self.z0_sz
		mu = torch.zeros(z0_sz)
		logsigma = torch.zeros(z0_sz)
		if device is not None:
			mu = mu.to(device)
			logsigma = logsigma.to(device)
		dist = Normal(mu, logsigma, temp=t)
		z, _ = dist.sample()
		latents = [z]

		idx = 0
		s = self.prior_ftr0.unsqueeze(0)
		s = s.expand(z.size(0), -1, -1, -1)
		for cell in self.dec_tower:
			if isinstance(cell, CombinerDec):
				if idx > 0:
					# form prior
					param = self.dec_sampler[idx - 1](s)
					mu, logsigma = torch.chunk(param, 2, dim=1)
					dist = Normal(mu, logsigma, t)
					z, _ = dist.sample()
					latents.append(z)

				# 'combiner_dec'
				s = cell(s, self.expand[idx](z))
				idx += 1
			else:
				s = cell(s)

		if self.vanilla:
			s = self.stem_decoder(z)

		for cell in self.post:
			s = cell(s)

		return self.out(s), latents

	@staticmethod
	def loss_kl(all_q, all_p, all_log_q, all_log_p):
		kl_all, kl_diag = [], []
		tot_log_q, tot_log_p = 0., 0.
		for q, p, log_q, log_p in zip(all_q, all_p, all_log_q, all_log_p):
			kl_per_var = q.kl(p)
			# if not self.cfg.compress:
			kl_per_var = torch.sum(kl_per_var, dim=[2, 3])
			kl_diag.append(torch.mean(kl_per_var, dim=0))  # TODO: mean? or sum?
			kl_all.append(torch.sum(kl_per_var, dim=1))
			# if not self.cfg.compress:
			tot_log_q += torch.sum(log_q, dim=[1, 2, 3])
			tot_log_p += torch.sum(log_q, dim=[1, 2, 3])
			# else:
			# tot_log_q += torch.sum(log_q, dim=1)
			# tot_log_p += torch.sum(log_q, dim=1)
		return tot_log_q, tot_log_p, kl_all, kl_diag

	def loss_spectral(self, device: torch.device = None):
		weights = collections.defaultdict(list)
		for lay in self.all_conv_layers:
			w = lay.w.view(lay.w.size(0), -1)
			weights[w.size()].append(w)
		weights = {
			k: torch.stack(v) for
			k, v in weights.items()
		}
		sr_loss = 0
		for i, w in weights.items():
			with torch.no_grad():
				n_iter = self.cfg.n_power_iter
				if i not in self.sr_u:
					num, row, col = w.size()
					self.sr_u[i] = F.normalize(
						torch.ones(num, row).normal_(0, 1).to(device),
						dim=1, eps=1e-3,
					)
					self.sr_v[i] = F.normalize(
						torch.ones(num, col).normal_(0, 1).to(device),
						dim=1, eps=1e-3,
					)
					# increase the number of iterations for the first time
					n_iter = 100 * self.cfg.n_power_iter

				for j in range(n_iter):
					# Spectral norm of weight equals to `u^T W v`, where `u` and `v`
					# are the first left and right singular vectors.
					# This power iteration produces approximations of `u` and `v`.
					self.sr_v[i] = F.normalize(
						torch.matmul(self.sr_u[i].unsqueeze(1), w).squeeze(1),
						dim=1, eps=1e-3,
					)  # bx1xr * bxrxc --> bx1xc --> bxc
					self.sr_u[i] = F.normalize(
						torch.matmul(w, self.sr_v[i].unsqueeze(2)).squeeze(2),
						dim=1, eps=1e-3,
					)  # bxrxc * bxcx1 --> bxrx1  --> bxr
			sigma = torch.matmul(
				self.sr_u[i].unsqueeze(1),
				torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)),
			)
			sr_loss += torch.sum(sigma)
		return sr_loss

	def loss_weight(self):
		return torch.abs(torch.cat(self.all_log_norm)).sum()

	def _init(self):
		self.vanilla = (
				self.cfg.n_latent_scales == 1 and
				self.cfg.n_groups_per_scale == 1
		)
		self.kws = dict(
			act_fn=self.cfg.activation_fn,
			use_bn=self.cfg.use_bn,
			use_se=self.cfg.use_se,
			normalize_dim=0,
		)
		self._init_stem()
		self._init_sizes()
		mult = self._init_pre(1)
		if not self.vanilla:
			mult = self._init_enc(mult)
		else:
			self.enc_tower = []
		mult = self._init_enc0(mult)
		self._init_sampler(mult)
		if not self.vanilla:
			mult = self._init_dec(mult)
			self.stem_decoder = None
		else:
			self.dec_tower = []
			self.stem_decoder = Conv2D(
				normalize_dim=self.kws['normalize_dim'],
				in_channels=self.cfg.n_latent_per_group,
				out_channels=int(mult * self.n_ch),
				kernel_size=1,
			)
		mult = self._init_post(mult)
		self._init_normalization()
		self._init_output(mult)

		return

	def _init_sizes(self):
		self.n_ch = self.cfg.n_kers * self.cfg.n_rots
		input_sz = self.cfg.input_sz - self.cfg.ker_sz + 1
		self.scales = [
			input_sz // _MULT ** (i + self.cfg.n_pre_blocks)
			for i in range(self.cfg.n_latent_scales)
		]
		self.z0_sz = [self.cfg.n_latent_per_group]
		self.z0_sz += [1 if self.cfg.compress else self.scales[-1]] * 2
		prior_ftr0_sz = [
			input_sz // self.scales[-1] * self.n_ch,
			self.scales[-1],
			self.scales[-1],
		]
		self.prior_ftr0 = nn.Parameter(
			data=torch.rand(prior_ftr0_sz),
			requires_grad=True,
		)
		return

	def _init_stem(self):
		if self.cfg.rot_equiv:
			self.stem = RotConv2d(
				co=self.cfg.n_kers,
				n_rots=self.cfg.n_rots,
				kernel_size=self.cfg.ker_sz,
				bias=True,
			)
		else:
			self.stem = Conv2D(
				in_channels=2,
				out_channels=self.cfg.n_kers * self.cfg.n_rots,
				normalize_dim=self.kws['normalize_dim'],
				kernel_size=self.cfg.ker_sz,
				padding='valid',
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
					enc.append(CombinerEnc(ch, ch))

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
		kws = dict(
			normalize_dim=self.kws['normalize_dim'],
			in_channels=ch,
			out_channels=ch,
			kernel_size=1,
			padding=0,
		)
		self.enc0 = nn.Sequential(
			nn.ELU(inplace=True),
			Conv2D(**kws),
			nn.ELU(inplace=True),
		)
		return mult

	def _init_sampler(self, mult):
		co = 2 * self.cfg.n_latent_per_group
		kws = dict(
			normalize_dim=1,
			out_channels=co,
		)
		enc_sampler = nn.ModuleList()
		dec_sampler = nn.ModuleList()
		expand = nn.ModuleList()
		for s in range(self.cfg.n_latent_scales):
			s_inv = self.cfg.n_latent_scales - s - 1
			ch = int(self.n_ch * mult)
			kws['in_channels'] = ch
			for g in range(self.cfg.groups[s_inv]):
				if self.cfg.compress:
					expand.append(DeConv2D(
						in_channels=self.cfg.n_latent_per_group,
						out_channels=self.cfg.n_latent_per_group,
						kernel_size=self.scales[s_inv],
						normalize_dim=0,
					))
					# expand.append(Expand(
					# 	normalize=self.kws['normalize'],
					# 	zdim=self.cfg.n_latent_per_group,
					# 	sdim=self.scales[s_inv],
					# ))
					kws['kernel_size'] = self.scales[s_inv]
					kws['padding'] = 0
				else:
					expand.append(nn.Identity())
					kws['kernel_size'] = 3
					kws['padding'] = 1
				enc_sampler.append(Conv2D(**kws))
				# enc_sampler.append(Conv2D(**kws))
				# expand.append(nn.Identity())
				if s == 0 and g == 0:
					continue  # 1st group: we used a fixed standard Normal
				if self.cfg.compress:
					kws['kernel_size'] = self.scales[s_inv]
					kws['padding'] = 0
					# dec_sampler.append(nn.Sequential(
					# 	nn.ELU(inplace=True),
					# 	Linear(**kws_lin),
					# ))
				else:
					kws['kernel_size'] = 1
					kws['padding'] = 0
				dec_sampler.append(nn.Sequential(
					nn.ELU(inplace=True),
					Conv2D(**kws),
				))
			mult /= _MULT
		self.enc_sampler = enc_sampler
		self.dec_sampler = dec_sampler
		self.expand = expand
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
				dec.append(CombinerDec(
					ci1=ch,
					ci2=self.cfg.n_latent_per_group,
					co=ch,
				))

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
		upsample = nn.Upsample(
				size=self.cfg.input_sz,
				mode='bilinear',
				align_corners=True,
			)
		looper = itertools.product(
			range(self.cfg.n_post_blocks),
			range(self.cfg.n_post_cells),
		)
		for b, c in looper:
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
			if c == 0 and b + 1 == self.cfg.n_post_blocks:
				post.append(upsample)
		if not len(post):
			post.append(upsample)
		self.post = post
		return mult

	def _init_output(self, mult):
		self.out = nn.Sequential(
			nn.ELU(inplace=True),
			nn.Conv2d(
				in_channels=int(self.n_ch * mult),
				out_channels=2,
				kernel_size=3,
				padding=1),
		)
		return

	def _init_normalization(self):
		self.all_log_norm = []
		self.all_conv_layers = []
		for n, lay in self.named_modules():
			if isinstance(lay, (Conv2D, DeConv2D)):
				self.all_log_norm.append(lay.log_weight_norm)
				self.all_conv_layers.append(lay)
		self.sr_u, self.sr_v = {}, {}
		if not self.cfg.spectral_reg:
			fn = AddNorm(
				norm='spectral',
				types=(nn.Conv2d, nn.ConvTranspose2d),
				n_power_iterations=self.cfg.n_power_iter,
			).get_fn()
			self.apply(fn)
		return


class Expand(nn.Module):
	def __init__(self, zdim: int, sdim: int, normalize: bool):
		super(Expand, self).__init__()
		self.linear = Linear(
			in_features=zdim,
			out_features=zdim * sdim ** 2,
			normalize=normalize,
		)
		self.shape = (-1, zdim, sdim, sdim)

	def forward(self, z):
		z = self.linear(z.squeeze())
		z = z.view(self.shape)
		return z


class CombinerEnc(nn.Module):
	def __init__(
			self,
			ci: int,
			co: int,
	):
		super(CombinerEnc, self).__init__()
		self.conv = Conv2D(
			in_channels=ci,
			out_channels=co,
			normalize_dim=0,
			kernel_size=1,
		)

	def forward(self, x1, x2):
		return x1 + self.conv(x2)


class CombinerDec(nn.Module):
	def __init__(
			self,
			ci1: int,
			ci2: int,
			co: int,
	):
		super(CombinerDec, self).__init__()
		self.conv = Conv2D(
			in_channels=ci1+ci2,
			out_channels=co,
			normalize_dim=0,
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
			use_se: bool,
			**kwargs,
	):
		super(Cell, self).__init__()

		self.skip = get_skip_connection(ci, _MULT, cell_type)
		self.ops = nn.ModuleList()
		for i in range(n_nodes):
			op = ConvLayer(
				ci=ci if i == 0 else co,
				co=co,
				stride=get_stride(cell_type, _MULT)
				if i == 0 else 1,
				act_fn=act_fn,
				use_bn=use_bn,
				**kwargs,
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
