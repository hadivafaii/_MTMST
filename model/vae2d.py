from .common import *
from .distributions import Normal


class VAE(Module):
	def __init__(self, cfg: ConfigVAE, **kwargs):
		super(VAE, self).__init__(cfg, **kwargs)
		self._init()
		if self.verbose:
			self.print()

	def forward(self, x):
		s = self.stem(x)

		for cell in self.pre_process:
			s = cell(s)

		# run the main encoder tower
		comb_enc, comb_s = [], []
		for cell in self.enc_tower:
			if isinstance(cell, CombinerEnc):
				comb_enc.append(cell)
				comb_s.append(s)
			else:
				s = cell(s)

		# reverse combiner cells and their input for decoder
		comb_enc.reverse()
		comb_s.reverse()

		idx = 0
		ftr_enc0 = self.enc0(s)
		param0 = self.enc_sampler[idx](ftr_enc0)
		mu_q, logsig_q = torch.chunk(param0, 2, dim=1)
		dist = Normal(mu_q, logsig_q)  # first approx. posterior
		z = dist.sample()
		q_all = [dist]
		latents = [z]

		# prior for z0
		dist = Normal(
			mu=torch.zeros_like(z),
			logsig=torch.zeros_like(z),
		)
		p_all = [dist]

		# begin decoder pathway
		s = self.prior_ftr0.unsqueeze(0)
		s = s.expand(z.size(0), -1, -1, -1)
		for cell in self.dec_tower:
			if isinstance(cell, CombinerDec):
				if idx > 0:
					# form prior
					param = self.dec_sampler[idx - 1](s)
					mu_p, logsig_p = torch.chunk(param, 2, dim=1)
					dist = Normal(mu_p, logsig_p)
					p_all.append(dist)

					# form encoder
					param = comb_enc[idx - 1](comb_s[idx - 1], s)
					param = self.enc_sampler[idx](param)
					mu_q, logsig_q = torch.chunk(param, 2, dim=1)
					if self.cfg.residual_kl:
						dist = Normal(
							mu=mu_q + mu_p,
							logsig=logsig_q + logsig_p,
						)
					else:
						dist = Normal(
							mu=mu_q,
							logsig=logsig_q,
						)
					q_all.append(dist)
					z = dist.sample()
					latents.append(z)
				# 'combiner_dec'
				s = cell(s, self.expand[idx](z))
				idx += 1
			else:
				s = cell(s)

		if self.vanilla:
			s = self.stem_decoder(z)

		for cell in self.post_process:
			s = cell(s)

		return self.out(s), latents, q_all, p_all

	@torch.no_grad()
	def sample(
			self,
			n: int = 1024,
			t: float = 1.0,
			device: torch.device = None, ):
		kws = dict(
			temp=t,
			device=device,
			seed=self.cfg.seed,
		)
		z0_sz = [n] + self.z0_sz
		mu = torch.zeros(z0_sz)
		logsig = torch.zeros(z0_sz)
		if device is not None:
			mu = mu.to(device)
			logsig = logsig.to(device)
		dist = Normal(mu, logsig, **kws)
		z = dist.sample()
		p_all = [dist]
		latents = [z]

		idx = 0
		s = self.prior_ftr0.unsqueeze(0)
		s = s.expand(z.size(0), -1, -1, -1)
		for cell in self.dec_tower:
			if isinstance(cell, CombinerDec):
				if idx > 0:
					# form prior
					param = self.dec_sampler[idx - 1](s)
					mu, logsig = torch.chunk(param, 2, dim=1)
					dist = Normal(mu, logsig, **kws)
					p_all.append(dist)
					z = dist.sample()
					latents.append(z)
				# 'combiner_dec'
				s = cell(s, self.expand[idx](z))
				idx += 1
			else:
				s = cell(s)

		if self.vanilla:
			s = self.stem_decoder(z)

		for cell in self.post_process:
			s = cell(s)

		return self.out(s), latents, p_all

	@torch.no_grad()
	def xtract_ftr(self, x, t: float = 0, full: bool = False):
		ftr_pre = collections.defaultdict(list)
		ftr_enc = collections.defaultdict(list)
		ftr_dec = collections.defaultdict(list)
		ftr_enc_sampler = collections.defaultdict(list)
		ftr_dec_sampler = collections.defaultdict(list)
		kws = dict(
			temp=t,
			device=x.device,
			seed=self.cfg.seed,
		)
		# enc
		s = self.stem(x)
		for cell in self.pre_process:
			s = cell(s)
			ftr_pre[s.size(-1)].append(s)
		comb_enc, comb_s = [], []
		for i, cell in enumerate(self.enc_tower):
			if isinstance(cell, CombinerEnc):
				comb_enc.append(cell)
				comb_s.append(s)
			else:
				s = cell(s)
				if full:
					ftr_enc[s.size(-1)].append(s)
		comb_enc.reverse()
		comb_s.reverse()

		idx = 0
		ftr_enc0 = self.enc0(s)
		param0 = self.enc_sampler[idx](ftr_enc0)
		mu_q, logsig_q = torch.chunk(param0, 2, dim=1)
		dist = Normal(mu_q, logsig_q, **kws)
		z = dist.sample()
		q_all = [dist]
		latents = [z]

		dist = Normal(
			mu=torch.zeros_like(z),
			logsig=torch.zeros_like(z),
			**kws,
		)
		p_all = [dist]

		# dec
		s = self.prior_ftr0.unsqueeze(0)
		s = s.expand(z.size(0), -1, -1, -1)
		for cell in self.dec_tower:
			if isinstance(cell, CombinerDec):
				if idx > 0:
					# form prior
					param = self.dec_sampler[idx - 1](s)
					if full:
						d = param.size(-1)
						ftr_dec_sampler[d].append(param)
					mu_p, logsig_p = torch.chunk(param, 2, dim=1)
					dist = Normal(mu_p, logsig_p, **kws)
					p_all.append(dist)

					# form encoder
					param = comb_enc[idx - 1](comb_s[idx - 1], s)
					param = self.enc_sampler[idx](param)
					if full:
						d = param.size(-1)
						ftr_enc_sampler[d].append(param)
					mu_q, logsig_q = torch.chunk(param, 2, dim=1)
					if self.cfg.residual_kl:
						dist = Normal(
							mu=mu_q + mu_p,
							logsig=logsig_q + logsig_p,
							**kws,
						)
					else:
						dist = Normal(
							mu=mu_q,
							logsig=logsig_q,
							**kws,
						)
					q_all.append(dist)
					z = dist.sample()
					latents.append(z)
				# 'combiner_dec'
				s = cell(s, self.expand[idx](z))
				idx += 1
			else:
				s = cell(s)
				if full:
					ftr_dec[s.size(-1)].append(s)

		if self.vanilla:
			s = self.stem_decoder(z)
			if full:
				ftr_dec[s.size(-1)].append(s)
		for cell in self.post_process:
			s = cell(s)
		ftr = {
			'enc0': ftr_enc0,
			'pre': dict(ftr_pre),
			'enc': dict(ftr_enc),
			'dec': dict(ftr_dec),
			'enc_sampler': dict(ftr_enc_sampler),
			'dec_sampler': dict(ftr_dec_sampler),
		}
		return self.out(s), latents, q_all, p_all, ftr

	def latent_scales(self):
		scales = itertools.chain.from_iterable([
			[scale] * num for num, scale in
			zip(self.cfg.groups, self.scales)
		])
		scales = list(scales)
		scales.reverse()
		# level ids
		idx, lvl_ids = 0, {}
		for s, c in collections.Counter(scales).items():
			for g in range(c):
				start = idx * self.cfg.n_latent_per_group
				stop = start + self.cfg.n_latent_per_group
				key = f"idx-{idx}_scale-{s}_group-{g}"
				lvl_ids[key] = range(start, stop)
				idx += 1
		return scales, lvl_ids

	def loss_recon(self, x, y, w=None):
		epe = endpoint_error(x, y)
		if self.cfg.balanced_recon:
			if w is None:
				w = torch.sum(torch.linalg.norm(
					x, dim=1), dim=[1, 2]).pow(-1)
			epe = epe * w / w.mean()
		return epe

	@staticmethod
	def loss_kl(q_all, p_all):
		kl_all, kl_diag = [], []
		for q, p in zip(q_all, p_all):
			kl = q.kl(p)
			kl = torch.sum(kl, dim=[2, 3])
			kl_all.append(torch.sum(kl, dim=1))
			kl_diag.append(torch.mean(kl, dim=0))
		return kl_all, kl_diag

	def loss_weight(self):
		return self.l1_weight(
			torch.cat(self.all_log_norm), self.w_tgt)

	def loss_spectral(self, device: torch.device = None, name: str = 'w'):
		weights = collections.defaultdict(list)
		for layer in self.all_conv_layers:
			w = getattr(layer, name)
			if isinstance(layer, DeConv2D):
				w = w.transpose(0, 1)  # 0: out_channels
				w = w.reshape(w.size(0), -1)
			elif isinstance(layer, Conv2D):
				w = w.view(w.size(0), -1)
			else:
				raise NotImplementedError()
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

				for _ in range(n_iter):
					# Spectral norm of weight equals to 'u^T W v', where 'u' and 'v'
					# are the first left and right singular vectors.
					# This power iteration produces approximations of 'u' and 'v'.
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

	def _init(self):
		self.vanilla = (
			self.cfg.n_latent_scales == 1 and
			self.cfg.n_groups_per_scale == 1
		)
		self.kws = dict(
			act_fn=self.cfg.activation_fn,
			use_bn=self.cfg.use_bn,
			use_se=self.cfg.use_se,
			scale=1.0,
		)
		self._init_stem()
		self._init_sizes()
		mult, depth = self._init_pre(1, 1)
		if not self.vanilla:
			mult = self._init_enc(mult, depth)
		else:
			self.enc_tower = []
		mult = self._init_enc0(mult)
		self._init_sampler(mult)
		self.kws['scale'] = 1.0
		if not self.vanilla:
			mult = self._init_dec(mult)
			self.stem_decoder = None
		else:
			self.dec_tower = []
			if self.cfg.compress:
				raise NotImplementedError
			else:
				self.stem_decoder = Conv2D(
					in_channels=self.cfg.n_latent_per_group,
					out_channels=int(mult * self.n_ch),
					kernel_size=1,
				)
		mult = self._init_post(mult)
		self._init_output(mult)
		self._init_norm()
		self._init_loss()
		return

	def _init_sizes(self):
		self.n_ch = self.cfg.n_kers * self.cfg.n_rots
		input_sz = self.cfg.input_sz - self.cfg.ker_sz + 1
		self.scales = [
			input_sz // MULT ** (i + self.cfg.n_pre_blocks)
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
				kernel_size=self.cfg.ker_sz,
				padding='valid',
			)
		return

	def _init_pre(self, mult, depth):
		pre = nn.ModuleList()
		looper = itertools.product(
			range(self.cfg.n_pre_blocks),
			range(self.cfg.n_pre_cells),
		)
		for _, c in looper:
			ch = int(self.n_ch * mult)
			if self.cfg.scale_init:
				self.kws['scale'] = 1 / np.sqrt(depth)
			if c == self.cfg.n_pre_cells - 1:
				co = MULT * ch
				cell = Cell(
					ci=ch,
					co=co,
					n_nodes=self.cfg.n_enc_nodes,
					cell_type='down_pre',
					**self.kws,
				)
				mult *= MULT
			else:
				cell = Cell(
					ci=ch,
					co=ch,
					n_nodes=self.cfg.n_enc_nodes,
					cell_type='normal_pre',
					**self.kws,
				)
			pre.append(cell)
			depth += 1
		self.pre_process = pre
		return mult, depth

	def _init_enc(self, mult, depth):
		enc = nn.ModuleList()
		for s in range(self.cfg.n_latent_scales):
			ch = int(self.n_ch * mult)
			for g in range(self.cfg.groups[s]):
				for _ in range(self.cfg.n_enc_cells):
					if self.cfg.scale_init:
						self.kws['scale'] = 1 / np.sqrt(depth)
					enc.append(Cell(
						ci=ch,
						co=ch,
						n_nodes=self.cfg.n_enc_nodes,
						cell_type='normal_pre',
						**self.kws,
					))
					depth += 1
				# add encoder combiner
				combiner = not (
					g == self.cfg.groups[s] - 1 and
					s == self.cfg.n_latent_scales - 1
				)
				if combiner:
					enc.append(CombinerEnc(ch, ch))

			# down cells after finishing a scale
			if s < self.cfg.n_latent_scales - 1:
				if self.cfg.scale_init:
					self.kws['scale'] = 1 / np.sqrt(depth)
				cell = Cell(
					ci=ch,
					co=ch * MULT,
					n_nodes=self.cfg.n_enc_nodes,
					cell_type='down_enc',
					**self.kws,
				)
				enc.append(cell)
				mult *= MULT
				depth += 1
		self.enc_tower = enc
		return mult

	def _init_enc0(self, mult):
		ch = int(self.n_ch * mult)
		kws = dict(
			in_channels=ch,
			out_channels=ch,
			kernel_size=1,
		)
		self.enc0 = nn.Sequential(
			nn.ELU(inplace=True),
			Conv2D(**kws),
			nn.ELU(inplace=True),
		)
		return mult

	def _init_sampler(self, mult):
		kws = dict(
			compress=self.cfg.compress,
			separable=self.cfg.separable,
			latent_dim=self.cfg.n_latent_per_group,
		)
		expand = nn.ModuleList()
		enc_sampler = nn.ModuleList()
		dec_sampler = nn.ModuleList()
		for s in range(self.cfg.n_latent_scales):
			s_inv = self.cfg.n_latent_scales - s - 1
			kws['spatial_dim'] = self.scales[s_inv]
			kws['in_channels'] = int(self.n_ch * mult)
			for g in range(self.cfg.groups[s_inv]):
				if self.cfg.compress:
					expand.append(DeConv2D(
						in_channels=self.cfg.n_latent_per_group,
						out_channels=self.cfg.n_latent_per_group,
						kernel_size=self.scales[s_inv],
						normalize_dim=1,
					))
				else:
					expand.append(nn.Identity())
				# enc sampler
				kws['act_fn'] = 'none'
				kws['init_scale'] = 0.7
				enc_sampler.append(Sampler(**kws))
				# dec sampler
				if s == 0 and g == 0:
					continue  # 1st group: we used a fixed standard Normal
				kws['act_fn'] = 'elu'
				kws['init_scale'] = 1e-2
				dec_sampler.append(Sampler(**kws))
			mult /= MULT
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
					for _ in range(self.cfg.n_dec_cells):
						dec.append(Cell(
							ci=ch,
							co=ch,
							n_nodes=self.cfg.n_dec_nodes,
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
					co=int(ch / MULT),
					n_nodes=self.cfg.n_dec_nodes,
					cell_type='up_dec',
					**self.kws,
				))
				mult /= MULT
		self.dec_tower = dec
		return mult

	def _init_post(self, mult):
		post = nn.ModuleList()
		upsample = nn.Upsample(
			size=self.cfg.input_sz,
			mode='nearest',
		)
		looper = itertools.product(
			range(self.cfg.n_post_blocks),
			range(self.cfg.n_post_cells),
		)
		for b, c in looper:
			ch = int(self.n_ch * mult)
			if c == 0:
				co = int(ch / MULT)
				cell = Cell(
					ci=ch,
					co=co,
					n_nodes=self.cfg.n_dec_nodes,
					cell_type='up_post',
					**self.kws,
				)
				mult /= MULT
			else:
				cell = Cell(
					ci=ch,
					co=ch,
					n_nodes=self.cfg.n_dec_nodes,
					cell_type='normal_post',
					**self.kws,
				)
			post.append(cell)
			if c == 0 and b + 1 == self.cfg.n_post_blocks:
				post.append(upsample)
		if not len(post):
			post.append(upsample)
		self.post_process = post
		return mult

	def _init_output(self, mult):
		kws = dict(
			in_channels=int(self.n_ch * mult),
			out_channels=2,
			kernel_size=3,
			padding=1,
		)
		self.out = nn.Conv2d(**kws)
		return

	def _init_norm(self, apply_norm: List[str] = None):
		apply_norm = apply_norm if apply_norm else [
			'stem', 'pre_process',
			'enc0', 'enc_tower', 'dec_tower',
			'enc_sampler', 'dec_sampler', 'expand',
		]
		self.all_log_norm = []
		self.all_conv_layers = []
		for child_name, child in self.named_children():
			for m in child.modules():
				if isinstance(m, (Conv2D, DeConv2D)):
					self.all_conv_layers.append(m)
					if child_name in apply_norm and m.apply_norm:
						self.all_log_norm.append(m.lognorm)
		self.sr_u, self.sr_v = {}, {}
		if self.cfg.spectral_norm:
			fn = AddNorm(
				norm='spectral',
				types=(nn.Conv2d, nn.ConvTranspose2d),
				n_power_iterations=self.cfg.spectral_norm,
				name='weight',
			).get_fn()
			self.apply(fn)
		return

	def _init_loss(self):
		self.l1_weight = nn.SmoothL1Loss(
			beta=0.1, reduction='mean')
		w_tgt = torch.zeros_like(
			torch.cat(self.all_log_norm))
		self.register_buffer('w_tgt', w_tgt)
		return


class Sampler(nn.Module):
	def __init__(
			self,
			in_channels: int,
			latent_dim: int,
			spatial_dim: int,
			act_fn: str = 'none',
			compress: bool = True,
			separable: bool = False,
			bias: bool = True,
			**kwargs,
	):
		super(Sampler, self).__init__()
		kws = dict(
			in_channels=in_channels,
			out_channels=latent_dim * 2,
			apply_norm=False,
			init_scale=1.0,
			bias=bias,
		)
		if compress:
			kws['kernel_size'] = spatial_dim
			kws['padding'] = 0
		else:
			kws['kernel_size'] = 3
			kws['padding'] = 1
		kwargs = setup_kwargs(kws, kwargs)

		if separable:
			depthwise = Conv2D(
				in_channels=in_channels,
				out_channels=in_channels,
				kernel_size=spatial_dim,
				groups=in_channels,
				apply_norm=True,
				bias=bias,
			)
			kwargs['kernel_size'] = 1
			pointwise = Conv2D(**kwargs)
			self.conv = nn.Sequential(
				depthwise, pointwise)
		else:
			self.conv = Conv2D(**kwargs)
		self.act = get_act(act_fn)

	def forward(self, x):
		if self.act is not None:
			x = self.act(x)
		x = self.conv(x)
		return x


class CombinerEnc(nn.Module):
	def __init__(
			self,
			ci: int,
			co: int,
	):
		super(CombinerEnc, self).__init__()
		self.conv = nn.Conv2d(
			in_channels=ci,
			out_channels=co,
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
		self.conv = nn.Conv2d(
			in_channels=ci1+ci2,
			out_channels=co,
			kernel_size=1,
		)

	def forward(self, x1, x2):
		x = torch.cat([x1, x2], dim=1)
		return self.conv(x)
