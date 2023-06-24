from .ae2d import AE
from base.train_base import *
from base.dataset import ROFLDS
from analysis.linear import regress, mi_analysis
from figures.fighelper import plot_heatmap, plot_bar


class TrainerAE(BaseTrainer):
	def __init__(
			self,
			model: AE,
			cfg: ConfigTrainAE,
			ema: bool = True,
			**kwargs,
	):
		super(TrainerAE, self).__init__(
			model=model, cfg=cfg, **kwargs)
		if ema:
			self.model_ema = AE(model.cfg).to(self.device).eval()
			self.ema_rate = self.to(self.cfg.ema_rate)
		self.n_iters = self.cfg.epochs * len(self.dl_trn)
		if self.cfg.lambda_anneal:
			self.wd_coeffs = beta_anneal_linear(
				n_iters=self.n_iters,
				beta=self.cfg.lambda_norm,
				min_beta=self.cfg.lambda_init,
			)
		else:
			self.wd_coeffs = np.ones(self.n_iters)
			self.wd_coeffs *= self.cfg.lambda_norm

	def iteration(self, epoch: int = 0, **kwargs):
		self.model.train()
		grads = AvgrageMeter()
		loss_meter = AvgrageMeter()
		perdim_epe = AvgrageMeter()
		for i, (x, norm) in enumerate(self.dl_trn):
			gstep = epoch * len(self.dl_trn) + i
			# warm-up lr
			if gstep < kwargs['n_iters_warmup']:
				lr = self.cfg.lr * gstep / kwargs['n_iters_warmup']
				for param_group in self.optim.param_groups:
					param_group['lr'] = lr
			# send to device
			if x.device != self.device:
				x, norm = self.to([x, norm])
			# zero grad
			self.optim.zero_grad(set_to_none=True)
			# forward + loss
			with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
				y, _ = self.model(x)
				epe = self.model.loss_recon(
					x=x, y=y, w=1/norm)
				loss = torch.mean(epe)
				# add regularization
				loss_w = self.model.loss_weight()
				if self.wd_coeffs[gstep] > 0:
					loss += self.wd_coeffs[gstep] * loss_w
				cond_reg_spectral = self.cfg.lambda_norm > 0 \
					and self.cfg.spectral_reg and \
					not self.model.cfg.spectral_norm
				if cond_reg_spectral:
					loss_sr = self.model.loss_spectral(
						device=self.device, name='w')
					loss += self.wd_coeffs[gstep] * loss_sr
				else:
					loss_sr = None
			# backward
			self.scaler.scale(loss).backward()
			self.scaler.unscale_(self.optim)
			# clip grad
			if self.cfg.grad_clip is not None:
				if gstep < kwargs['n_iters_warmup']:
					max_norm = self.cfg.grad_clip * 2
				else:
					max_norm = self.cfg.grad_clip
				grad_norm = nn.utils.clip_grad_norm_(
					parameters=self.model.parameters(),
					max_norm=max_norm,
				).item()
				grads.update(grad_norm)
				self.stats['grad'][gstep] = grad_norm
				if grad_norm > self.cfg.grad_clip:
					self.stats['loss'][gstep] = loss.item()
			# update average meters & stats
			loss_meter.update(loss.item())
			perdim_epe.update(epe.mean().item() / self.model.cfg.input_sz ** 2)
			msg = [
				f"gstep # {gstep:.3g}",
				f"loss: {loss_meter.avg:0.3f}",
			]
			if self.cfg.grad_clip:
				msg += [f"grad: {grads.val:0.1f}"]
			self.pbar.set_description(', '.join(msg))
			# step
			self.scaler.step(self.optim)
			self.scaler.update()
			self.update_ema()
			# optim schedule
			cond_schedule = (
				gstep > kwargs['n_iters_warmup']
				and self.optim_schedule is not None
			)
			if cond_schedule:
				self.optim_schedule.step()
			# write
			cond_write = (
				gstep > 0 and
				self.writer is not None and
				gstep % self.cfg.log_freq == 0
			)
			if not cond_write:
				continue
			to_write = {
				'ae/reg_coeff': self.wd_coeffs[gstep],
				'ae/lr': self.optim.param_groups[0]['lr'],
				'ae/perdim_epe': perdim_epe.avg,
				'ae/reg_weight': loss_w.item(),
			}
			if self.cfg.grad_clip is not None:
				to_write['ae/grad_norm'] = grads.avg
			if cond_reg_spectral:
				to_write['ae/reg_spectral'] = loss_sr.item()
			for k, v in to_write.items():
				self.writer.add_scalar(k, v, gstep)
			# reset average meters
			if gstep % (self.cfg.log_freq * 10) == 0:
				loss_meter.reset()
				grads.reset()

		return loss_meter.avg

	def validate(
			self,
			gstep: int = None,
			use_ema: bool = False, ):
		data, loss = self.forward('vld', use_ema=use_ema)
		regr = self.regress(use_ema=use_ema)

		# plot?
		if gstep is not None:
			freq = max(20, self.cfg.eval_freq * 5)
			ep = int(gstep / len(self.dl_trn))
			cond = ep % freq == 0
		else:
			cond = True
		if cond:
			mi, figs = self.plot(
				regr=regr,
				use_ema=use_ema,
			)
			data = {**mi, **figs}
		else:
			mi, figs = None, None
		# write
		if gstep is not None:
			to_write = {
				f"ae/{k}": v.mean()
				for k, v in loss.items()
			}
			to_write = {
				**to_write,
				'ae/r': np.diag(regr['regr/r']).mean(),
				'ae/r2': np.nanmean(regr['regr/r2']) * 100,
				'ae/r_aux': np.diag(regr['regr/aux/r']).mean(),
				'ae/r2_aux': np.nanmean(regr['regr/aux/r2']) * 100,
				'ae/disentang': regr['regr/d'],
				'ae/complete': regr['regr/c'],
			}
			for k, v in to_write.items():
				self.writer.add_scalar(k, v, gstep)
				self.stats[k][gstep] = v
			if cond:
				to_write = {
					'ae/mi': np.max(mi['regr/mi'], 1).mean(),
					'ae/mi_norm': np.max(mi['regr/mi_norm'], 1).mean(),
					'ae/mig': mi['regr/mig'].mean(),
				}
				for k, v in to_write.items():
					self.writer.add_scalar(k, v, gstep)
					self.stats[k][gstep] = v
				for k, v in figs.items():
					self.writer.add_figure(k, v, gstep)
		return data, loss

	def forward(
			self,
			dl_name: str,
			use_ema: bool = False, ):
		assert dl_name in ['trn', 'vld', 'tst']
		dl = getattr(self, f"dl_{dl_name}")
		if dl is None:
			return
		model = self.select_model(use_ema)

		epe = []
		x_all, y_all, h_all = [], [], []
		for i, (x, norm) in enumerate(dl):
			if x.device != self.device:
				x, norm = self.to([x, norm])
			h, _, y = model.xtract_ftr(x)
			h = torch.cat(h, dim=1).squeeze()
			# data
			if dl_name == 'trn':
				x_all.append(to_np(x))
				y_all.append(to_np(y))
			h_all.append(to_np(h))
			# loss
			epe.append(to_np(model.loss_recon(
				x=x, y=y, w=1 / norm)))

		x, y, h, epe = cat_map(
			[x_all, y_all, h_all, epe])
		data = {'x': x, 'y': y, 'z': h}
		loss = {'epe': epe}
		return data, loss

	def regress(self, use_ema: bool = False):
		z_vld = self.forward('vld', use_ema)[0]['z']
		z_tst = self.forward('tst', use_ema)[0]['z']
		regr = regress(
			z=z_vld,
			z_tst=z_tst,
			g=self.dl_vld.dataset.g,
			g_tst=self.dl_tst.dataset.g,
		)
		regr_aux = regress(
			z=z_vld,
			z_tst=z_tst,
			g=self.dl_vld.dataset.g_aux,
			g_tst=self.dl_tst.dataset.g_aux,
		)
		regr = {
			f"regr/{k}": v for
			k, v in regr.items()
		}
		regr_aux.update({
			f"regr/aux/{k}": v for
			k, v in regr_aux.items()
		})
		output = {
			'z_vld': z_vld,
			'z_tst': z_tst,
			**regr,
			**regr_aux,
		}
		return output

	def plot(self, regr: dict = None, **kwargs):
		regr = regr if regr else self.regress(
			**filter_kwargs(self.regress, kwargs))

		figs = {}

		# corr (regression)
		f = self.dl_tst.dataset.f
		_tx = [f"({i:02d})" for i in range(len(f))]
		_ty = [f"{e} ({i:02d})" for i, e in enumerate(f)]
		rd = np.diag(regr['regr/r'])
		title = f"all  =  {rd.mean():0.3f} ± {rd.std():0.3f}  "
		title += r'$(\mu \pm \sigma)$' + '\n'
		name_groups = collections.defaultdict(list)
		for i, lbl in enumerate(f):
			k = lbl.split('_')[0]
			name_groups[k].append(i)
		for i, (k, ids) in enumerate(name_groups.items()):
			title += f"{k} :  {rd[ids].mean():0.2f},"
			title += ' ' * 5
			if (i + 1) % 3 == 0:
				title += '\n'
		fig, _ = plot_heatmap(
			r=regr['regr/r'],
			title=title,
			cmap='PiYG',
			xticklabels=_tx,
			yticklabels=_ty,
			annot_kws={'fontsize': 12},
			figsize=(0.72 * len(f), 0.6 * len(f)),
			display=False,
		)
		figs['fig/regression'] = fig

		# barplots
		df = pd.DataFrame({
			'x': self.dl_vld.dataset.f,
			'y': regr['regr/r2'],
		})
		fig, _ = plot_bar(df, tick_labelsize_x=10, display=False)
		figs['fig/bar'] = fig
		# aux
		df = pd.DataFrame({
			'x': self.dl_vld.dataset.f_aux,
			'y': regr['regr/aux/r2'],
		})
		fig, _ = plot_bar(df, tick_labelsize_x=10, display=False)
		figs['fig/bar_aux'] = fig

		# mutual info
		n_jobs = max(1, joblib.effective_n_jobs())
		n_jobs /= max(1, torch.cuda.device_count())
		mi = mi_analysis(
			z=regr['z_vld'],
			g=self.dl_vld.dataset.g,
			n_jobs=int(n_jobs),
		)
		mi = {
			f"regr/{k}": v for
			k, v in mi.items()
		}
		regr = {**regr, **mi}
		title = '_'.join(self.model.cfg.name().split('_')[:3])
		mi_max = np.round(np.max(regr['regr/mi'], axis=1), 2)
		mi_max = ', '.join([str(e) for e in mi_max])
		title = f"model = {title};    max MI (row) = {mi_max}"
		figsize = (0.08 * self.model.cfg.total_latents(), 0.72 * len(f))
		fig, _ = plot_heatmap(
			r=regr['regr/mi'],
			yticklabels=_ty,
			title=title,
			tick_labelsize_x=10,
			tick_labelsize_y=7,
			title_fontsize=14,
			title_y=1.02,
			vmin=0,
			vmax=0.65,
			cmap='rocket',
			linecolor='dimgrey',
			figsize=figsize,
			cbar=False,
			annot=False,
			display=False,
		)
		figs['fig/mutual_info'] = fig
		return regr, figs

	def setup_data(self, gpu: bool = True):
		# create datasets
		device = self.device if gpu else None
		ds_trn = ROFLDS(self.model.cfg.sim_path, 'trn', device)
		ds_vld = ROFLDS(self.model.cfg.sim_path, 'vld', device)
		ds_tst = ROFLDS(self.model.cfg.sim_path, 'tst', device)
		# cleate dataloaders
		kws = dict(
			batch_size=self.cfg.batch_size,
			drop_last=True,
			shuffle=True,
		)
		self.dl_trn = DataLoader(ds_trn, **kws)
		kws.update({'drop_last': False, 'shuffle': False})
		self.dl_vld = DataLoader(ds_vld, **kws)
		self.dl_tst = DataLoader(ds_tst, **kws)
		return

	def reset_model(self):
		self.model = AE(self.model.cfg).to(self.device)
		self.model_ema = AE(self.model.cfg).to(self.device)
		return


def _setup_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"sim",
		help='simulation category',
		type=str,
	)
	parser.add_argument(
		"device",
		help='cuda:n',
		type=str,
	)
	parser.add_argument(
		"--comment",
		help='comment',
		default=None,
		type=str,
	)
	parser.add_argument(
		"--n_ch",
		help='# channels',
		default=32,
		type=int,
	)
	parser.add_argument(
		"--input_sz",
		help='ROFL dim',
		default=33,
		type=int,
	)
	parser.add_argument(
		"--res_eps",
		help='x + eps * f(x)',
		default=0.1,
		type=float,
	)
	# enc
	parser.add_argument(
		"--n_enc_cells",
		help='# enc cells',
		default=1,
		type=int,
	)
	parser.add_argument(
		"--n_enc_nodes",
		help='# enc nodes',
		default=3,
		type=int,
	)
	# dec
	parser.add_argument(
		"--n_dec_cells",
		help='# dec cells',
		default=1,
		type=int,
	)
	parser.add_argument(
		"--n_dec_nodes",
		help='# dec nodes',
		default=2,
		type=int,
	)
	# pre
	parser.add_argument(
		"--n_pre_cells",
		help='# preprocessing cells',
		default=3,
		type=int,
	)
	parser.add_argument(
		"--n_pre_blocks",
		help='# preprocessing blocks',
		default=1,
		type=int,
	)
	# post
	parser.add_argument(
		"--n_post_cells",
		help='# postprocessing cells',
		default=3,
		type=int,
	)
	parser.add_argument(
		"--n_post_blocks",
		help='# postprocessing blocks',
		default=1,
		type=int,
	)
	# latents
	parser.add_argument(
		"--n_latent_scales",
		help='# latent scales',
		default=3,
		type=int,
	)
	parser.add_argument(
		"--n_latent_per_group",
		help='# latents per group',
		default=20,
		type=int,
	)
	parser.add_argument(
		"--n_groups_per_scale",
		help='# groups per scale',
		default=12,
		type=int,
	)
	parser.add_argument(
		"--activation_fn",
		help='activation function',
		default='swish',
		type=str,
	)
	parser.add_argument(
		"--spectral_norm",
		help='spectral norm (0 = disable)',
		default=0,
		type=int,
	)
	parser.add_argument(
		"--ada_groups",
		help='adaptive latent groups?',
		default=True,
		type=true_fn,
	)
	parser.add_argument(
		"--compress",
		help='compress latent space?',
		default=True,
		type=true_fn,
	)
	parser.add_argument(
		"--use_bn",
		help='use batch norm?',
		default=False,
		type=true_fn,
	)
	parser.add_argument(
		"--use_se",
		help='use squeeze & excite?',
		default=True,
		type=true_fn,
	)
	# training
	parser.add_argument(
		"--lr",
		help='learning rate',
		default=0.002,
		type=float,
	)
	parser.add_argument(
		"--epochs",
		help='# epochs',
		default=160,
		type=int,
	)
	parser.add_argument(
		"--batch_size",
		help='batch size',
		default=600,
		type=int,
	)
	parser.add_argument(
		"--warm_restart",
		help='# warm restarts',
		default=0,
		type=int,
	)
	parser.add_argument(
		"--warmup_portion",
		help='warmup portion',
		default=1.25e-2,
		type=float,
	)
	parser.add_argument(
		"--optimizer",
		help='optimizer',
		default='adamax_fast',
		type=str,
	)
	parser.add_argument(
		"--lambda_anneal",
		help='anneal weight reg coeff?',
		default=False,
		type=true_fn,
	)
	parser.add_argument(
		"--lambda_norm",
		help='weight regularization strength',
		default=1e-4,
		type=float,
	)
	parser.add_argument(
		"--grad_clip",
		help='gradient norm clipping',
		default=350.0,
		type=float,
	)
	parser.add_argument(
		"--seed",
		help='random seed',
		default=0,
		type=int,
	)
	parser.add_argument(
		"--chkpt_freq",
		help='checkpoint freq',
		default=10,
		type=int,
	)
	parser.add_argument(
		"--eval_freq",
		help='eval freq',
		default=2,
		type=int,
	)
	parser.add_argument(
		"--log_freq",
		help='log freq',
		default=10,
		type=int,
	)
	parser.add_argument(
		"--dry_run",
		help='to make sure config is alright',
		action='store_true',
		default=False,
	)
	return parser.parse_args()


def _main():
	args = _setup_args()
	print(args)

	ae = AE(ConfigAE(
		sim=args.sim,
		seed=args.seed,
		n_ch=args.n_ch,
		res_eps=args.res_eps,
		input_sz=args.input_sz,
		n_enc_cells=args.n_enc_cells,
		n_enc_nodes=args.n_enc_nodes,
		n_dec_cells=args.n_dec_cells,
		n_dec_nodes=args.n_dec_nodes,
		n_pre_cells=args.n_pre_cells,
		n_pre_blocks=args.n_pre_blocks,
		n_post_cells=args.n_post_cells,
		n_post_blocks=args.n_post_blocks,
		n_latent_scales=args.n_latent_scales,
		n_latent_per_group=args.n_latent_per_group,
		n_groups_per_scale=args.n_groups_per_scale,
		activation_fn=args.activation_fn,
		spectral_norm=args.spectral_norm,
		ada_groups=args.ada_groups,
		compress=args.compress,
		save=not args.dry_run,
		use_bn=args.use_bn,
		use_se=args.use_se,
		balanced_recon=True,
		scale_init=False,
		separable=False,
	))
	tr = TrainerAE(
		model=ae,
		device=args.device,
		cfg=ConfigTrainAE(
			lr=args.lr,
			epochs=args.epochs,
			batch_size=args.batch_size,
			warm_restart=args.warm_restart,
			warmup_portion=args.warmup_portion,
			optimizer=args.optimizer,
			grad_clip=args.grad_clip,
			# weight reg
			lambda_anneal=args.lambda_anneal,
			lambda_norm=args.lambda_norm,
			lambda_init=1e-7,
			# freqs
			chkpt_freq=args.chkpt_freq,
			eval_freq=args.eval_freq,
			log_freq=args.log_freq),
	)
	msg = ', '.join([
		f"# enc ftrs: {sum(ae.ftr_sizes()[0].values())}",
		f"# conv layers: {len(ae.all_conv_layers)}",
		f"# latents: {ae.cfg.total_latents()}",
	])
	print('\n', msg)
	ae.print()
	msg = '\n'.join([
		f"AE:\t\t{ae.cfg.name()}",
		f"Trainer:\t{tr.cfg.name()}\n",
	])
	print(msg)

	if args.comment is not None:
		comment = '_'.join([
			args.comment,
			tr.cfg.name(),
		])
	else:
		comment = tr.cfg.name()

	if not args.dry_run:
		tr.train(comment)

	print(f"\n[PROGRESS] fitting AE on {args.device} done ({now(True)}).\n")
	return


if __name__ == "__main__":
	_main()
