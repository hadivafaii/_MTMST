from utils.plotting import *
from base.dataset import ROFLDS
from analysis.helper import vel2polar
from matplotlib.gridspec import GridSpec


def get_betas(df: pd.DataFrame):
	betas: List[Union[float, str]] = ['ae']
	betas += sorted([
		b for b in df['beta'].unique()
		if isinstance(b, float)
	])
	return betas


def get_palette(pal: str = 'muted'):
	pal = sns.color_palette(pal)
	pal_model = {
		'cNVAE': pal[0],
		'NVAE': pal[9],
		'VAE': pal[1],
		'cNAE': '#6f6f6f',
		'AE': '#aeaeae',
		'PCA': pal[2],
		'Raw': pal[4],
	}
	pal_cat = {
		'obj1': pal[6],
		'fixate0': pal[9],
		'fixate1': pal[0],
		'fixate2': '#28529d',
		'transl1': pal[2],
	}
	return pal_model, pal_cat


def extract_info(fit: str):
	info = fit.split('_')
	i = info.index([
		e for e in info
		if 'nf-' in e
	].pop())
	category = info[i - 1]
	nf = int(info[i].split('-')[1])
	beta = info[i + 1].split('-')[1]
	try:
		beta = float(beta)
	except ValueError:
		beta = str(beta)
	model = 'VAE' if 'vanilla' in info else 'cNVAE'
	if beta == 'ae':
		model = model.replace('V', '')
	return category, nf, beta, model


def prep_rofl(
		cat: str = 'fixate1',
		labels: List[str] = None, ):
	sim_path = pjoin(
		os.environ['HOME'],
		'Documents/MTVAE/data',
		f"{cat}_dim-17_n-750k",
	)
	ds = {
		k: ROFLDS(sim_path, k) for
		k in ['trn', 'vld', 'tst']
	}
	# select main ground-truth variables
	f = ds['trn'].f + ds['trn'].f_aux
	labels = labels if labels else LBL2TEX
	select_i, select_lbl = zip(*[
		(i, lbl) for i, lbl
		in enumerate(f) if
		lbl in labels
	])
	perm = [
		select_lbl.index(lbl) for
		i, lbl in enumerate(labels)
		if lbl in select_lbl
	]
	select_i = np.array(select_i)[perm]
	select_lbl = np.array(select_lbl)[perm]
	select_lbl = list(select_lbl)
	g = {
		k: np.concatenate(
			[v.g, v.g_aux],
			axis=1)[:, select_i]
		for k, v in ds.items()
	}
	return g, select_i, select_lbl


def show_neural_results(
		df: pd.DataFrame,
		perf: str = 'perf',
		display: bool = True,
		**kwargs, ):
	defaults = dict(
		figsize=(8, 4),
	)
	kwargs = setup_kwargs(defaults, kwargs)
	fig, axes = create_figure(
		nrows=2,
		ncols=2,
		figsize=kwargs['figsize'],
		layout='constrained',
	)
	sns.histplot(
		data=df,
		x=perf,
		bins=np.linspace(0, 1, 41),
		label=r"$R$",
		ax=axes[0, 0],
	)
	x = np.mean(df[perf])
	axes[0, 0].axvline(
		x=x,
		ls='--',
		color='r',
		label=r"$\mu_R = $" + f"{x:0.3f}",
	)
	axes[0, 0].locator_params(
		axis='x', nbins=12)
	axes[0, 0].set(xlabel='')

	x = 'log_alpha'
	a, b = min(df[x]), max(df[x])
	bins = np.linspace(a, b + 1, int(b - a) + 2) - 0.5
	sns.histplot(
		x=x,
		data=df,
		bins=bins,
		label=r'$\log \alpha$',
		ax=axes[0, 1],
	)
	axes[0, 1].tick_params(
		axis='x', rotation=-90, labelsize=9)
	axes[0, 1].locator_params(
		axis='x', nbins=len(bins) + 2)
	axes[0, 1].set(ylabel='', xlabel='')

	x = 'best_lag'
	a, b = min(df[x]), max(df[x])
	bins = np.linspace(a, b + 1, int(b - a) + 2) - 0.5
	sns.histplot(
		x=x,
		data=df,
		bins=bins,
		label='lag used (in fit)',
		ax=axes[1, 0],
	)
	axes[1, 0].locator_params(
		axis='x', nbins=len(bins) + 2)
	axes[1, 0].set(xlabel='')

	x = 'r_tst_norm'
	if not all(np.isnan(df[x].values)):
		sns.histplot(
			x=x,
			data=df,
			bins=np.linspace(0, 1, 41),
			label=r"$R$ (test / nrm)",
			ax=axes[1, 1],
		)
		x = np.mean(df[x][df[x] > 0])
		axes[1, 1].axvline(
			x=x,
			ls='--',
			color='g',
			label=r"$\mu_R = $" + f"{x:0.3f}",
		)
		axes[1, 1].locator_params(
			axis='x', nbins=len(bins) + 2)
		axes[1, 1].set(xlabel='', ylabel='')

	for ax in axes.flat:
		ax.legend()

	if display:
		plt.show()
	else:
		plt.close()
	return fig, axes


def plot_bar(df: pd.DataFrame, display: bool = True, **kwargs):
	defaults = dict(
		x='x',
		y='y',
		figsize_y=7,
		figsize_x=0.7,
		tick_labelsize_x=15,
		tick_labelsize_y=15,
		ylabel_fontsize=20,
		title_fontsize=18,
		vals_fontsize=13,
		title_y=1,
	)
	kwargs = setup_kwargs(defaults, kwargs)
	figsize = (
		kwargs['figsize_x'] * len(df),
		kwargs['figsize_y'],
	)
	fig, ax = create_figure(1, 1, figsize)
	bp = sns.barplot(data=df, x=kwargs['x'], y=kwargs['y'], ax=ax)
	barplot_add_vals(bp, fontsize=kwargs['vals_fontsize'])
	ax.tick_params(
		axis='x',
		rotation=-90,
		labelsize=kwargs['tick_labelsize_x'],
	)
	ax.tick_params(
		axis='y',
		labelsize=kwargs['tick_labelsize_y'],
	)
	val = np.nanmean(df[kwargs['y']]) * 100
	title = r'avg $R^2 = $' + f"{val:0.1f} %"
	ax.set_title(
		label=title,
		y=kwargs['title_y'],
		fontsize=kwargs['title_fontsize'],
	)
	ax.set_ylabel(
		ylabel=r'$R^2$',
		fontsize=kwargs['ylabel_fontsize'],
	)
	ax.set(xlabel='', ylim=(0, 1))
	ax.grid()
	if display:
		plt.show()
	else:
		plt.close()
	return fig, ax


def plot_heatmap(
		r: np.ndarray,
		title: str = None,
		xticklabels: List[str] = None,
		yticklabels: List[str] = None,
		display: bool = True,
		**kwargs, ):
	defaults = dict(
		figsize=(10, 8),
		tick_labelsize_x=12,
		tick_labelsize_y=12,
		title_fontsize=13,
		title_y=1,
		vmin=-1,
		vmax=1,
		cmap='bwr',
		linewidths=0.005,
		linecolor='silver',
		square=True,
		annot=True,
		fmt='.1f',
		annot_kws={'fontsize': 8},
	)
	kwargs = setup_kwargs(defaults, kwargs)
	fig, ax = create_figure(figsize=kwargs['figsize'])
	sns.heatmap(r, ax=ax, **filter_kwargs(sns.heatmap, kwargs))
	if title is not None:
		ax.set_title(
			label=title,
			y=kwargs['title_y'],
			fontsize=kwargs['title_fontsize'],
		)
	if xticklabels is not None:
		ax.set_xticklabels(xticklabels)
		ax.tick_params(
			axis='x',
			rotation=-90,
			labelsize=kwargs['tick_labelsize_x'],
		)
	if yticklabels is not None:
		ax.set_yticklabels(yticklabels)
		ax.tick_params(
			axis='y',
			rotation=0,
			labelsize=kwargs['tick_labelsize_y'],
		)
	if display:
		plt.show()
	else:
		plt.close()
	return fig, ax


def plot_latents_hist_full(
		z: np.ndarray,
		scales: List[int],
		display: bool = True,
		**kwargs, ):
	defaults = dict(
		bins_divive=128,
		figsize=None,
		figsize_x=3.25,
		figsize_y=2.75,
		layout='tight',
	)
	kwargs = setup_kwargs(defaults, kwargs)

	a = z.reshape((len(z), len(scales), -1))
	nrows, ncols = a.shape[1:]
	if kwargs['figsize'] is not None:
		figsize = kwargs['figsize']
	else:
		figsize = (
			kwargs['figsize_x'] * ncols,
			kwargs['figsize_y'] * nrows,
		)
	fig, axes = create_figure(
		nrows=nrows,
		ncols=ncols,
		sharey='row',
		figsize=figsize,
		layout=kwargs['layout'],
	)
	looper = itertools.product(
		range(nrows), range(ncols))
	for i, j in looper:
		x = a[:, i, j]
		sns.histplot(
			x,
			stat='percent',
			bins=len(a) // kwargs['bins_divive'],
			ax=axes[i, j],
		)
		msg = r"$\mu = $" + f"{x.mean():0.2f}, "
		msg += r"$\sigma = $" + f"{x.std():0.2f}\n"
		msg += f"minmax = ({x.min():0.2f}, {x.max():0.2f})\n"
		msg += f"skew = {sp_stats.skew(x):0.2f}"
		axes[i, j].set_title(msg)
	add_grid(axes)
	if display:
		plt.show()
	else:
		plt.close()
	return fig, axes


def plot_latents_hist(
		z: np.ndarray,
		scales: List[int],
		display: bool = True,
		**kwargs, ):
	defaults = dict(
		bins_divive=64,
		figsize=None,
		figsize_x=3.25,
		figsize_y=2.75,
		layout='tight',
	)
	kwargs = setup_kwargs(defaults, kwargs)

	nrows = 2
	ncols = int(np.ceil(len(scales) / nrows))
	if kwargs['figsize'] is not None:
		figsize = kwargs['figsize']
	else:
		figsize = (
			kwargs['figsize_x'] * ncols,
			kwargs['figsize_y'] * nrows,
		)
	fig, axes = create_figure(
		nrows=nrows,
		ncols=ncols,
		figsize=figsize,
		layout=kwargs['layout'],
	)
	a = z.reshape((len(z), len(scales), -1))
	for i in range(len(scales)):
		ax = axes.flat[i]
		x = a[:, i, :].ravel()
		sns.histplot(
			x,
			label=f"s = {scales[i]}",
			bins=len(a)//kwargs['bins_divive'],
			stat='percent',
			ax=ax,
		)
		msg = r"$\mu = $" + f"{x.mean():0.2f}, "
		msg += r"$\sigma = $" + f"{x.std():0.2f}\n"
		msg += f"minmax = ({x.min():0.2f}, {x.max():0.2f})\n"
		msg += f"skew = {sp_stats.skew(x):0.2f}"
		ax.set_ylabel('')
		ax.set_title(msg)
		ax.legend(loc='upper right')
	for i in range(2):
		axes[i, 0].set_ylabel('Proportion [%]')
	trim_axs(axes, len(scales))
	add_grid(axes)
	if display:
		plt.show()
	else:
		plt.close()
	return fig, axes


def plot_opticflow_hist(
		x: np.ndarray,
		val: float = 1.0,
		display: bool = True,
		**kwargs, ):
	defaults = dict(
		figsize=(9.5, 4.5),
		layout='constrained',
		leg_fontsize=12,
		leg_loc='best',
	)
	kwargs = setup_kwargs(defaults, kwargs)
	rho, theta = vel2polar(x)
	fig, axes = create_figure(
		nrows=2,
		ncols=3,
		figsize=kwargs['figsize'],
		layout=kwargs['layout'],
	)
	sns.histplot(
		rho.ravel(),
		stat='percent',
		label=r'$\rho$',
		ax=axes[0, 0],
	)
	sns.histplot(
		rho.ravel(),
		stat='percent',
		label=r'$\rho$',
		ax=axes[0, 1],
	)
	sns.histplot(
		np.log(rho[rho.nonzero()]),
		stat='percent',
		label=r'$\log \, \rho$',
		ax=axes[0, 2],
	)
	sns.histplot(
		rho.mean(1).mean(1),
		stat='percent',
		label='norm',
		ax=axes[1, 0],
	)
	sns.histplot(
		theta.ravel(),
		stat='percent',
		label=r'$\theta$',
		ax=axes[1, 1],
	)
	sns.histplot(
		x.ravel(),
		stat='percent',
		label='velocity',
		ax=axes[1, 2],
	)
	for ax in axes.flat:
		ax.set_ylabel('')
		ax.legend(
			fontsize=kwargs['leg_fontsize'],
			loc=kwargs['leg_loc'],
		)
	for ax in axes[0, :2].flat:
		ax.axvline(val, color='r', ls='--', lw=1.2)
	axes[0, 2].axvline(np.log(val), color='r', ls='--', lw=1.2)
	axes[0, 0].set_ylabel('[%]', fontsize=15)
	axes[1, 0].set_ylabel('[%]', fontsize=15)
	axes[0, 1].set_yscale('log')
	axes[0, 2].set_xlim(-10, 2 + np.log(val))
	axes[1, 0].set_xlim(left=0)
	axes[1, 2].set_xlim(-val, val)
	if display:
		plt.show()
	else:
		plt.close()
	return fig, axes


def show_opticflow(
		x: np.ndarray,
		num: int = 4,
		titles: list = None,
		no_ticks: bool = True,
		display: bool = True,
		**kwargs, ):
	defaults = {
		'figsize': (7, 7),
		'tick_spacing': 4,
		'title_fontsize': 11,
		'layout': 'constrained',
		'scale': None,
	}
	kwargs = setup_kwargs(defaults, kwargs)
	x = to_np(x)
	if x.shape[-1] == 2:
		x = np.transpose(x, (0, -1, 1, 2))
	d, odd = x.shape[2] // 2, x.shape[2] % 2
	span = range(-d, d + 1) if odd else range(-d, d)
	ticks, ticklabels = make_ticks(
		span, kwargs['tick_spacing'])
	fig, axes = create_figure(
		nrows=num,
		ncols=num,
		sharex='all',
		sharey='all',
		figsize=kwargs['figsize'],
		layout=kwargs['layout'],
		reshape=True,
	)
	for i, ax in enumerate(axes.flat):
		try:
			v = x[i]
		except IndexError:
			ax.remove()
			continue
		if titles is not None:
			try:
				ax.set_title(
					label=titles[i],
					fontsize=kwargs['title_fontsize'],
				)
			except IndexError:
				pass
		ax.quiver(
			span, span, v[0], v[1],
			scale=kwargs['scale'],
		)
		ax.set(
			xticks=ticks,
			yticks=ticks,
			xticklabels=ticklabels,
			yticklabels=ticklabels,
		)
		ax.tick_params(labelsize=8)
	ax_square(axes)
	if no_ticks:
		remove_ticks(axes, False)
	if display:
		plt.show()
	else:
		plt.close()
	return fig, axes


def show_opticflow_full(
		v: np.ndarray,
		cbar: bool = False,
		display: bool = True,
		**kwargs, ):
	defaults = {
		'cmap_v': 'bwr',
		'cmap_rho': 'Spectral_r',
		'figsize': (7 if cbar else 5.25, 6.7),
		'title_fontsize': 9,
		'cbar_ticksize': 9,
		'tick_spacing': 4,
		'scale': None,
	}
	kwargs = setup_kwargs(defaults, kwargs)
	v = to_np(v)
	assert v.ndim == 3, "(2, x, y)"
	if v.shape[2] == 2:
		rho, phi = vel2polar(v)
		v = np.transpose(v, (2, 0, 1))
	else:
		rho, phi = vel2polar(np.transpose(v, (1, 2, 0)))
	d, odd = v.shape[1] // 2, v.shape[1] % 2
	span = range(-d, d + 1) if odd else range(-d, d)
	ticks, ticklabels = make_ticks(
		span, kwargs['tick_spacing'])
	vminmax = np.max(np.abs(v))
	kws_v = dict(
		vmax=vminmax,
		vmin=-vminmax,
		cmap=kwargs['cmap_v'],
	)
	gs = GridSpec(
		nrows=5,
		ncols=6 if cbar else 4,
		width_ratios=[1, 1, 0.1, 1, 0.1, 1] if cbar else None,
	)
	fig = plt.figure(figsize=kwargs['figsize'])
	axes = []

	ax1 = fig.add_subplot(gs[0, 0])
	ax1.imshow(v[0], **kws_v)
	title = r'$v_x \in $' + f"({kws_v['vmin']:0.1f},{kws_v['vmax']:0.1f})"
	ax1.set_title(label=title, y=1.02, fontsize=kwargs['title_fontsize'])
	axes.append(ax1)

	ax2 = fig.add_subplot(gs[0, 1])
	im = ax2.imshow(v[1], **kws_v)
	if cbar:
		cb = plt.colorbar(im, ax=ax2)
		cb.ax.tick_params(labelsize=kwargs['cbar_ticksize'])
	title = r'$v_y \in $' + f"({kws_v['vmin']:0.1f},{kws_v['vmax']:0.1f})"
	ax2.set_title(label=title, y=1.02, fontsize=kwargs['title_fontsize'])
	axes.append(ax2)

	ax3 = fig.add_subplot(gs[0, 3 if cbar else 2])
	im = ax3.imshow(rho, cmap=kwargs['cmap_rho'])
	if cbar:
		cb = plt.colorbar(im, ax=ax3)
		cb.ax.tick_params(labelsize=kwargs['cbar_ticksize'])
	title = r'$\rho \in $' + f"[{np.min(rho):0.1f},{np.max(rho):0.1f}]"
	ax3.set_title(label=title, y=1.02, fontsize=kwargs['title_fontsize'])
	axes.append(ax3)

	ax4 = fig.add_subplot(gs[0, 5 if cbar else 3])
	im = ax4.imshow(phi, cmap='hsv', vmin=0, vmax=2*np.pi)
	if cbar:
		cb = plt.colorbar(im, ax=ax4)
		cb.ax.tick_params(labelsize=kwargs['cbar_ticksize'])
	title = r'$\phi \in \left[0, 2\pi\right)$'
	ax4.set_title(label=title, y=1.02, fontsize=kwargs['title_fontsize'])
	axes.append(ax4)

	ax = fig.add_subplot(gs[1:, :])
	ax.quiver(
		span, span, v[0], v[1],
		scale=kwargs['scale'],
	)
	ax.set(
		xticks=ticks,
		yticks=ticks,
		xticklabels=ticklabels,
		yticklabels=ticklabels,
	)
	ax.tick_params(labelsize=13)
	axes.append(ax)
	axes = np.array(axes)
	ax_square(axes)
	for ax in axes[:-1]:
		ax.invert_yaxis()
		ax.set(
			xticks=[t + d for t in ticks],
			yticks=[t + d for t in ticks],
			xticklabels=[],
			yticklabels=[],
		)
	if display:
		plt.show()
	else:
		plt.close()
	return fig, axes


def show_opticflow_row(
		x: np.ndarray,
		titles: np.ndarray = None,
		scale: Sequence[float] = None,
		no_ticks: bool = True,
		display: bool = True,
		**kwargs, ):
	defaults = {
		'figsize': (9, 4),
		'tick_spacing': 4,
		'title_fontsize': 11,
		'layout': 'constrained',
		'height_ratios': None,
	}
	kwargs = setup_kwargs(defaults, kwargs)
	x = to_np(x)
	if x.shape[-1] == 2:
		x = np.transpose(x, (0, 1, -1, 2, 3))
	d, odd = x.shape[-2] // 2, x.shape[-2] % 2
	span = range(-d, d + 1) if odd else range(-d, d)
	ticks, ticklabels = make_ticks(
		span, kwargs['tick_spacing'])
	nrows = x.shape[0]
	ncols = x.shape[1]
	fig, axes = create_figure(
		nrows=nrows,
		ncols=ncols,
		sharex='all',
		sharey='all',
		layout=kwargs['layout'],
		figsize=kwargs['figsize'],
		height_ratios=kwargs['height_ratios'],
	)
	looper = itertools.product(
		range(nrows),
		range(ncols),
	)
	for i, j in looper:
		ax = axes[i, j]
		try:
			v = x[i, j]
		except IndexError:
			ax.remove()
			continue
		if titles is not None:
			try:
				ax.set_title(
					label=titles[i, j],
					fontsize=kwargs['title_fontsize'],
				)
			except IndexError:
				pass
		ax.quiver(
			span, span,
			v[0], v[1],
			scale=scale,
		)
		ax.set(
			xticks=ticks,
			yticks=ticks,
			xticklabels=ticklabels,
			yticklabels=ticklabels,
		)
		ax.tick_params(labelsize=8)
	if no_ticks:
		remove_ticks(axes, False)
	ax_square(axes)
	if display:
		plt.show()
	else:
		plt.close()
	return fig, axes


CAT2TEX = {
	'fixate0': '\\fixate{0}',
	'fixate1': '\\fixate{1}',
	'obj1': '\\obj{1}',
}

LBL2TEX = {
	'fix_x': r'$F_x$',
	'fix_y': r'$F_y$',
	'slf_v_x': r'$V_{self, x}$',
	'slf_v_y': r'$V_{self, y}$',
	'slf_v_z': r'$V_{self, z}$',
	'obj0_x': r'$X_{obj}$',
	'obj0_y': r'$Y_{obj}$',
	'obj0_z': r'$Z_{obj}$',
	'obj0_v_x': r'$V_{obj, x}$',
	'obj0_v_y': r'$V_{obj, y}$',
	'obj0_v_z': r'$V_{obj, z}$',
}


# TODO: later remove above and keep below
LBL2TEX_main = {
	'fix_x': r'$F_x$',
	'fix_y': r'$F_y$',
	'slf_v_x': r'$V_{self, x}$',
	'slf_v_y': r'$V_{self, y}$',
	'slf_v_z': r'$V_{self, z}$',
	'obj0_alpha_x': r'$X_{obj}$',
	'obj0_alpha_y': r'$Y_{obj}$',
	'obj0_size': r'$S_{obj}$',
	'obj0_v_x': r'$V_{obj, x}$',
	'obj0_v_y': r'$V_{obj, y}$',
	'obj0_v_z': r'$V_{obj, z}$',
}
