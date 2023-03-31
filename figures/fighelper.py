from utils.plotting import *
from analysis.helper import vel2polar
from matplotlib.gridspec import GridSpec


def show_heatmap(
		r: np.ndarray,
		title: str = None,
		xticklabels: List[str] = None,
		yticklabels: List[str] = None,
		display: bool = True,
		**kwargs, ):
	defaults = dict(
		figsize=(15, 12),
		tick_labelsize_x=14,
		tick_labelsize_y=14,
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
		tight_layout=True,
		constrained_layout=False,
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
		tight_layout=kwargs['tight_layout'],
		constrained_layout=kwargs['constrained_layout'],
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
		tight_layout=True,
		constrained_layout=False,
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
		tight_layout=kwargs['tight_layout'],
		constrained_layout=kwargs['constrained_layout'],
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
		display: bool = True, ):
	rho, theta = vel2polar(x)
	fig, axes = create_figure(
		nrows=2,
		ncols=3,
		figsize=(12, 6),
		constrained_layout=True,
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
		ax.legend(fontsize=15, loc='upper right')
	for ax in axes[0, :2].flat:
		ax.axvline(1, color='r', ls='--', lw=1.2)
	axes[0, 2].axvline(0, color='r', ls='--', lw=1.2)
	axes[0, 0].set_ylabel('[%]', fontsize=15)
	axes[1, 0].set_ylabel('[%]', fontsize=15)
	axes[0, 1].set_yscale('log')
	axes[0, 2].set_xlim(-10, 2)
	axes[1, 0].set_xlim(left=0)
	if display:
		plt.show()
	else:
		plt.close()
	return fig, axes


def show_opticflow(
		x: np.ndarray,
		num: int = 4,
		titles: list = None,
		display: bool = True,
		**kwargs, ):
	defaults = {
		'figsize': (9, 9),
		'tick_spacing': 3,
		'title_fontsize': 15,
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
		constrained_layout=True,
		tight_layout=False,
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
	if display:
		plt.show()
	else:
		plt.close()
	return fig, axes


def show_opticflow_full(v: np.ndarray, **kwargs):
	defaults = {
		'cmap_v': 'bwr',
		'cmap_rho': 'Spectral_r',
		'figsize': (10, 9),
		'tick_spacing': 3,
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
		ncols=5,
		width_ratios=[1, 1, 0.01, 1, 1],
	)
	fig = plt.figure(figsize=kwargs['figsize'])
	axes = []

	ax1 = fig.add_subplot(gs[0, 0])
	im = ax1.imshow(v[0], **kws_v)
	plt.colorbar(im, ax=ax1)
	ax1.set_title(r'$v_x$', y=1.02, fontsize=13)
	axes.append(ax1)

	ax2 = fig.add_subplot(gs[0, 1])
	im = ax2.imshow(v[1], **kws_v)
	plt.colorbar(im, ax=ax2)
	ax2.set_title(r'$v_y$', y=1.02, fontsize=13)
	axes.append(ax2)

	ax3 = fig.add_subplot(gs[0, 3])
	im = ax3.imshow(rho, cmap=kwargs['cmap_rho'])
	plt.colorbar(im, ax=ax3)
	ax3.set_title(r'$\rho$', y=1.02, fontsize=13)
	axes.append(ax3)

	ax4 = fig.add_subplot(gs[0, 4])
	im = ax4.imshow(phi, cmap='hsv', vmin=0, vmax=2*np.pi)
	plt.colorbar(im, ax=ax4)
	ax4.set_title(r'$\phi$', y=1.02, fontsize=12)
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
	plt.show()
	return fig, axes


def make_ticks(span, tick_spacing):
	ticks, ticklabels = zip(*[
		(x, (str(x))) for i, x
		in enumerate(span) if
		i % tick_spacing == 0
	])
	return ticks, ticklabels
