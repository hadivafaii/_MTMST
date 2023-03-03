from model.utils_model import *
from matplotlib.gridspec import GridSpec


def show_opticflow(y: np.ndarray, n: int = 4, **kwargs):
	defaults = {
		'figsize': (9, 9),
		'tick_spacing': 3,
	}
	kwargs = setup_kwargs(defaults, kwargs)
	y = to_np(y)
	d, odd = y.shape[2] // 2, y.shape[2] % 2
	span = range(-d, d + 1) if odd else range(-d, d)
	ticks, ticklabels = make_ticks(
		span, kwargs['tick_spacing'])
	fig, axes = create_figure(
		nrows=n,
		ncols=n,
		sharex='all',
		sharey='all',
		figsize=kwargs['figsize'],
		constrained_layout=True,
		tight_layout=False,
	)
	for i, ax in enumerate(axes.flat):
		try:
			v = y[i]
		except IndexError:
			ax.remove()
			continue
		ax.quiver(span, span, v[0], v[1])
		ax.set(
			xticks=ticks,
			yticks=ticks,
			xticklabels=ticklabels,
			yticklabels=ticklabels,
		)
		ax.tick_params(labelsize=8)
	ax_square(axes)
	return fig, axes


def show_opticflow_full(v: np.ndarray, **kwargs):
	defaults = {
		'cmap_v': 'bwr',
		'cmap_rho': 'Spectral_r',
		'figsize': (10, 9),
		'tick_spacing': 3,
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
	ax.quiver(span, span, v[0], v[1])
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
