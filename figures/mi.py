from .fighelper import *


def plot_hm(
		data: Dict[str, np.ndarray],
		scale_factor: int = 25,
		display: bool = True, ):
	assert scale_factor % 2 == 1, "must be an odd number"
	fig, axes = create_figure(
		nrows=2,
		ncols=1,
		figsize=(10, 4.5),
		sharex='all',
		sharey='all',
		layout='constrained',
	)
	for i, (fit, v) in enumerate(data.items()):
		x2p = np.repeat(np.repeat(
			v, scale_factor, axis=0),
			scale_factor, axis=1)
		axes[i].imshow(
			X=x2p,
			aspect=9,
			vmin=0,
			vmax=0.5,
			cmap='bone_r',
		)
	yticks = np.arange(
		start=0,
		stop=len(LBL2TEX) * scale_factor,
		step=scale_factor,
	) + scale_factor // 2
	xticks = np.arange(
		start=0,
		stop=420 * scale_factor + 1,
		step=20 * scale_factor,
	)
	axes[0].set(
		yticks=yticks,
		yticklabels=list(LBL2TEX.values()),
	)
	axes[1].set(
		yticks=yticks,
		xticks=xticks,
		yticklabels=list(LBL2TEX.values()),
		xticklabels=np.arange(0, 420 + 1, 20),
	)
	for i in range(1, 21):
		val = i * 20 - 0.5
		axes[0].axvline(
			x=val * scale_factor,
			color='dimgrey',
			alpha=0.7,
			ls='--',
			lw=0.7,
		)
	if display:
		plt.show()
	else:
		plt.close()
	return fig, axes
