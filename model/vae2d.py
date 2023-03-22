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

		for cell in self.pre:
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
		param0 = self.enc_sampler[idx](s)
		mu_q, logsig_q = torch.chunk(param0, 2, dim=1)
		dist = Normal(mu_q, logsig_q)  # for the first approx. posterior
		z = dist.sample()
		q_all = [dist]
		latents = [z]

		# prior for z0
		dist = Normal(
			mu=torch.zeros_like(z),
			logsigma=torch.zeros_like(z),
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
							logsigma=logsig_q + logsig_p,
						)
					else:
						dist = Normal(mu_q, logsig_q)
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

		for cell in self.post:
			s = cell(s)

		return self.out(s), self._cat(latents), q_all, p_all

	def sample(self, n: int, t: float = 1.0, device: torch.device = None):
		z0_sz = [n] + self.z0_sz
		mu = torch.zeros(z0_sz)
		logsigma = torch.zeros(z0_sz)
		if device is not None:
			mu = mu.to(device)
			logsigma = logsigma.to(device)
		dist = Normal(mu, logsigma, temp=t)
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
					mu, logsigma = torch.chunk(param, 2, dim=1)
					dist = Normal(mu, logsigma, t)
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

		for cell in self.post:
			s = cell(s)

		return self.out(s), self._cat(latents), p_all

	def _cat(self, latents):
		if self.cfg.compress:
			return torch.cat(latents, dim=1).squeeze()
		else:
			return latents

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
		l1 = self.l1_recon(x, y)
		l2 = self.l2_recon(x, y)
		l1 = torch.sum(l1, dim=[1, 2, 3])
		l2 = torch.sum(l2, dim=[1, 2, 3])
		epe = endpoint_error(x, y)
		if self.cfg.balanced_recon:
			if w is None:
				w = torch.sum(torch.linalg.norm(
					x, dim=1), dim=[1, 2]).pow(-1)
			l1 = l1 * w / w.mean()
			l2 = l2 * w / w.mean()
			epe = epe * w / w.mean()
		cos = 1 - self.cos_recon(x, y)
		cos = torch.sum(cos, dim=[1, 2])
		return l1, l2, epe, cos

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

				for j in range(n_iter):
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
		)
		self._init_stem()
		self._init_sizes()
		mult, depth = self._init_pre(1, 1)
		if not self.vanilla:
			mult = self._init_enc(mult, depth)
		else:
			self.enc_tower = []
		self._init_sampler(mult)
		if not self.vanilla:
			mult, depth = self._init_dec(mult, 1)
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
		mult = self._init_post(mult, depth)
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
			if c == self.cfg.n_pre_cells - 1:
				co = MULT * ch
				cell = Cell(
					ci=ch,
					co=co,
					n_nodes=self.cfg.n_enc_nodes,
					init_scale=1/np.sqrt(depth) if
					self.cfg.scale_init else None,
					cell_type='down_pre',
					**self.kws,
				)
				mult *= MULT
			else:
				cell = Cell(
					ci=ch,
					co=ch,
					n_nodes=self.cfg.n_enc_nodes,
					init_scale=1/np.sqrt(depth) if
					self.cfg.scale_init else None,
					cell_type='normal_pre',
					**self.kws,
				)
			pre.append(cell)
			depth += 1
		self.pre = pre
		return mult, depth

	def _init_enc(self, mult, depth):
		enc = nn.ModuleList()
		for s in range(self.cfg.n_latent_scales):
			ch = int(self.n_ch * mult)
			for g in range(self.cfg.groups[s]):
				for _ in range(self.cfg.n_cells_per_cond):
					enc.append(Cell(
						ci=ch,
						co=ch,
						n_nodes=self.cfg.n_enc_nodes,
						init_scale=1/np.sqrt(depth) if
						self.cfg.scale_init else None,
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
				cell = Cell(
					ci=ch,
					co=ch * MULT,
					n_nodes=self.cfg.n_enc_nodes,
					init_scale=1/np.sqrt(depth) if
					self.cfg.scale_init else None,
					cell_type='down_enc',
					**self.kws,
				)
				enc.append(cell)
				mult *= MULT
				depth += 1
		self.enc_tower = enc
		return mult

	def _init_sampler(self, mult):
		co = 2 * self.cfg.n_latent_per_group
		kws = dict(out_channels=co)
		enc_sampler = nn.ModuleList()
		dec_sampler = nn.ModuleList()
		expand = nn.ModuleList()
		for s in range(self.cfg.n_latent_scales):
			s_inv = self.cfg.n_latent_scales - s - 1
			ch = int(self.n_ch * mult)
			kws['in_channels'] = ch
			for g in range(self.cfg.groups[s_inv]):
				if self.cfg.compress:
					expand.append(nn.ConvTranspose2d(
						in_channels=self.cfg.n_latent_per_group,
						out_channels=self.cfg.n_latent_per_group,
						kernel_size=self.scales[s_inv],
					))
					kws['kernel_size'] = self.scales[s_inv]
					kws['padding'] = 0
				else:
					expand.append(nn.Identity())
					kws['kernel_size'] = 3
					kws['padding'] = 1
				enc_sampler.append(nn.Conv2d(**kws))
				if s == 0 and g == 0:
					continue  # 1st group: we used a fixed standard Normal
				if self.cfg.compress:
					kws['kernel_size'] = self.scales[s_inv]
					kws['padding'] = 0
				else:
					kws['kernel_size'] = 1
					kws['padding'] = 0
				dec_sampler.append(nn.Sequential(
					nn.ELU(inplace=True),
					nn.Conv2d(**kws),
				))
			mult /= MULT
		self.enc_sampler = enc_sampler
		self.dec_sampler = dec_sampler
		self.expand = expand
		return

	def _init_dec(self, mult, depth):
		dec = nn.ModuleList()
		for s in range(self.cfg.n_latent_scales):
			ch = int(self.n_ch * mult)
			for g in range(self.cfg.groups[self.cfg.n_latent_scales - s - 1]):
				if not (s == 0 and g == 0):
					for _ in range(self.cfg.n_cells_per_cond):
						dec.append(Cell(
							ci=ch,
							co=ch,
							n_nodes=self.cfg.n_dec_nodes,
							init_scale=1/np.sqrt(depth) if
							self.cfg.scale_init else None,
							cell_type='normal_dec',
							**self.kws,
						))
						depth += 1
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
					init_scale=1/np.sqrt(depth) if
					self.cfg.scale_init else None,
					cell_type='up_dec',
					**self.kws,
				))
				mult /= MULT
				depth += 1
		self.dec_tower = dec
		return mult, depth

	def _init_post(self, mult, depth):
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
					init_scale=1/np.sqrt(depth) if
					self.cfg.scale_init else None,
					cell_type='up_post',
					**self.kws,
				)
				mult /= MULT
			else:
				cell = Cell(
					ci=ch,
					co=ch,
					n_nodes=self.cfg.n_dec_nodes,
					init_scale=1/np.sqrt(depth) if
					self.cfg.scale_init else None,
					cell_type='normal_post',
					**self.kws,
				)
			post.append(cell)
			depth += 1
			if c == 0 and b + 1 == self.cfg.n_post_blocks:
				post.append(upsample)
		if not len(post):
			post.append(upsample)
		self.post = post
		return mult

	def _init_output(self, mult):
		self.out = nn.Conv2d(
			in_channels=int(self.n_ch * mult),
			out_channels=2,
			kernel_size=3,
			padding=1)
		return

	def _init_norm(self, apply_norm: List[str] = None):
		apply_norm = apply_norm if apply_norm else [
			'stem', 'pre', 'enc_tower', 'dec_tower']
		self.all_log_norm = []
		self.all_conv_layers = []
		for child_name, child in self.named_children():
			for m in child.modules():
				if isinstance(m, (Conv2D, DeConv2D)):
					self.all_conv_layers.append(m)
					if child_name in apply_norm:
						self.all_log_norm.append(
							m.log_weight_norm)
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
		self.cos_recon = nn.CosineSimilarity(dim=1)
		self.l2_recon = nn.MSELoss(reduction='none')
		self.l1_recon = nn.SmoothL1Loss(
			beta=0.1, reduction='none')
		self.l1_weight = nn.SmoothL1Loss(
			beta=0.1, reduction='mean')
		w_tgt = torch.zeros_like(
			torch.cat(self.all_log_norm))
		self.register_buffer('w_tgt', w_tgt)
		return


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
