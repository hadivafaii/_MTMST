from .plotting import *
from matplotlib import animation


def animate_opticflow(
		x: np.ndarray,
		save_file: str,
		no_ticks: bool = True,
		display: bool = False,
		nrows: int = 1,
		**kwargs, ):
	defaults = {
		'scale': None,
		'figscale': 2.4,
		'figsize': None,
		'tick_spacing': 4,
		'title_fontsize': 11,
		'layout': 'constrained',
		'title': 'opticflow animation',
		'artist': 'Hadi Vafaii',
		'interval': 100,
		'bitrate': -1,
		'dpi': 600,
		'fps': 15,
	}
	kwargs = setup_kwargs(defaults, kwargs)
	# data has shape N x t x 2 x dim x dim
	x = to_np(x)
	if x.ndim == 4:
		x = np.expand_dims(x, 0)
	elif x.ndim == 5:
		pass
	else:
		raise ValueError(x.ndim)
	if x.shape[-1] == 2:
		x = np.transpose(x, (0, 1, -1, 2, 3))
	n, frames, _, _, d = x.shape
	d, odd = x.shape[-2] // 2, x.shape[-2] % 2
	span = range(-d, d + 1) if odd else range(-d, d)
	ticks, ticklabels = make_ticks(
		span, kwargs['tick_spacing'])
	ncols = int(np.ceil(n / nrows))

	if kwargs['figsize'] is None:
		figsize = (
			ncols * kwargs['figscale'],
			nrows * kwargs['figscale'],
		)
	else:
		figsize = kwargs['figsize']
	fig, axes = create_figure(
		nrows=nrows,
		ncols=ncols,
		sharex='all',
		sharey='all',
		reshape=True,
		figsize=figsize,
		layout=kwargs['layout'],
	)
	plots = ()
	looper = itertools.product(
		range(nrows),
		range(ncols),
	)
	for i, j in looper:
		ax = axes[i, j]
		idx = i * ncols + j
		if idx >= n:
			ax.remove()
			continue
		u, v = x[idx, 0, 0, ...], x[idx, 0, 1, ...]
		vel = ax.quiver(
			span, span, u, v,
			scale=kwargs['scale'],
		)
		ax.set(
			xticks=ticks,
			yticks=ticks,
			xticklabels=ticklabels,
			yticklabels=ticklabels,
		)
		plots += (vel,)
	if no_ticks:
		remove_ticks(axes, False)
	ax_square(axes)

	def _update(t):
		_looper = itertools.product(
			range(nrows),
			range(ncols),
		)
		for _i, _j in _looper:
			_idx = _i * ncols + _j
			if _idx >= n:
				continue
			plots[_idx].set_UVC(
				x[_idx, t, 0, ...],
				x[_idx, t, 1, ...],
			)
		return plots

	anim = animation.FuncAnimation(
		fig=fig,
		func=_update,
		frames=frames,
		interval=kwargs['interval'],
		blit=True,
	)
	writer = animation.FFMpegWriter(
		fps=kwargs['fps'],
		metadata={
			'title': kwargs['title'],
			'artist': kwargs['artist']},
		bitrate=kwargs['bitrate'],
	)
	anim.save(
		filename=save_file,
		dpi=kwargs['dpi'],
		writer=writer,
	)
	if display:
		plt.show(fig)
	else:
		plt.close(fig)
	return anim, writer
