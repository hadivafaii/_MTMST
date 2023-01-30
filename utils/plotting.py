from .generic import *
import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt
# from matplotlib.gridspec import GridSpec
# from matplotlib.patches import PathPatch
# from matplotlib.path import Path as mpl_Path
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages
from matplotlib.colors import to_rgb, rgb2hex, Colormap, LinearSegmentedColormap


# noinspection PyUnresolvedReferences
def show_hsv(figsize=(3, 3), num=2048):
	fig = plt.figure(figsize=figsize)
	cax = fig.add_axes([0, 0, 1, 1], projection='polar')
	cbar = matplotlib.colorbar.ColorbarBase(
		ax=cax,
		cmap=matplotlib.cm.get_cmap('hsv', num),
		norm=matplotlib.colors.Normalize(0, 2*np.pi),
		orientation='horizontal',
	)
	cax.get_children()[1].set_lw(0)
	cax.axis('off')
	plt.show()
	return fig, cax, cbar


# noinspection PyUnresolvedReferences
def cbar_only(
		cmap,
		vmin: float = 0,
		vmax: float = 1,
		vertical: bool = True,
		**kwargs, ):
	defaults = {
		'figsize': (0.4, 4) if
		vertical else (4, 0.35),
		'edgecolor': 'k',
		'linewidth': 1.3,
		'tick_pad': 2,
		'tick_length': 6,
		'tick_labelsize': 12,
		'tick_position': 'right'
		if vertical else 'bottom',
	}
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v

	fig, cax = create_figure(
		nrows=1,
		ncols=1,
		figsize=kwargs['figsize'],
		constrained_layout=False,
		tight_layout=False,
	)
	cbar = matplotlib.colorbar.ColorbarBase(
		ax=cax,
		cmap=matplotlib.cm.get_cmap(cmap)
		if isinstance(cmap, str) else cmap,
		norm=matplotlib.colors.Normalize(vmin, vmax),
		orientation='vertical' if vertical else 'horizontal',
	)

	cbar.outline.set_edgecolor(kwargs['edgecolor'])
	cbar.outline.set_linewidth(kwargs['linewidth'])

	cax.tick_params(
		axis='y' if vertical else 'x',
		pad=kwargs['tick_pad'],
		length=kwargs['tick_length'],
		labelsize=kwargs['tick_labelsize'],
		color=kwargs['edgecolor'],
		width=kwargs['linewidth'],
	)
	if vertical:
		cax.yaxis.set_ticks_position(kwargs['tick_position'])
	else:
		cax.xaxis.set_ticks_position(kwargs['tick_position'])
	plt.close()
	return fig, cax, cbar


def add_jitter(
		x: np.ndarray,
		sigma: float = 0.01,
		shift_mean: bool = True, ):
	jit = get_rng().normal(scale=sigma, size=len(x))
	if shift_mean:
		jit -= jit.mean()
	return x + jit


def _iter_ax(axes):
	if not isinstance(axes, Iterable):
		return [axes]
	elif isinstance(axes, np.ndarray):
		return axes.flat


def ax_square(axes):
	for ax in _iter_ax(axes):
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		delta_x = max(xlim) - min(xlim)
		delta_y = max(ylim) - min(ylim)
		aspect = delta_x / delta_y
		ax.set_aspect(aspect, adjustable='box')
	return axes


def trim_axs(axes, n):
	axs = axes.flat
	for ax in axs[n:]:
		ax.remove()
	return axs[:n]


def add_grid(axes):
	for ax in _iter_ax(axes):
		ax.grid()
	return axes


def add_ax_inset(
		ax: plt.Axes,
		data: np.ndarray,
		bounds: List[float],
		kind: str = 'imshow',
		aspect_eq: bool = True,
		**kwargs, ):
	axins = ax.inset_axes(bounds)
	if kind == 'imshow':
		x2p = np.ma.masked_where(data[0] == 0, data[0])
		axins.imshow(x2p, cmap='Greys_r')
		x2p = np.ma.masked_where(data[1] < kwargs['vmin'], data[1])
		axins.imshow(x2p, **kwargs)
	elif kind == 'kde':
		sns.kdeplot(data=data, ax=axins, **kwargs)
	else:
		raise NotImplementedError
	if aspect_eq:
		axins.set_aspect('equal', adjustable='box')
	return axins


def remove_ticks(axes, full=True):
	for ax in _iter_ax(axes):
		ax.set_xticks([])
		ax.set_yticks([])
		if full:
			try:
				_ = list(map(
					lambda z: z.set_visible(False),
					ax.spines.values()
				))
			except AttributeError:
				continue
	return


def make_cmap(
		ramp_colors: List[str],
		name: str = 'custom_cmap',
		n_colors: int = 256,
		show: bool = True, ):
	color_ramp = LinearSegmentedColormap.from_list(
		name=name,
		colors=[to_rgb(c) for c in ramp_colors],
		N=n_colors,
	)
	if show:
		display_cmap(color_ramp, len(ramp_colors))
	return color_ramp


def get_rand_cmap(num: int, rng=None):
	rng = rng if rng else get_rng()
	colors = (
		rng.choice(256, size=num) / 256,
		rng.choice(256, size=num) / 256,
		rng.choice(256, size=num) / 256,
	)
	colors = [
		rgb2hex(c) for c in
		list(zip(*colors))
	]
	cmap = make_cmap(
		ramp_colors=colors,
		name='random',
		n_colors=num,
		show=False,
	)
	return cmap


def get_hm_cmap(
		colors: List[str] = None,
		return_clist: bool = False, ):
	colors = colors if colors else [
		'#365c7f', '#09f6e1', '#7ac837',
		'#fbf504', '#f0640f', '#f50a10',
	]
	heatmap = make_cmap(
		ramp_colors=colors,
		name='heatmap',
		show=False,
	)
	if return_clist:
		return heatmap, colors
	else:
		return heatmap


def display_cmap(cmap: Union[str, Colormap], num_colors: int = 256):
	plt.figure(figsize=(13.5, 3))
	plt.imshow(
		[list(np.arange(0, num_colors, 0.11 * num_colors / 4))],
		interpolation='nearest',
		origin='lower',
		cmap=cmap,
	)
	plt.xticks([])
	plt.yticks([])
	plt.tight_layout()
	plt.show()
	return


def cmap2pal(cmap: Colormap, n_colors: int, start: float):
	x = np.linspace(start % 1, 1.0, n_colors)
	x = [cmap(abs(i - 1e-10)) for i in x]
	return sns.color_palette(x)


def complement_color(r, g, b):
	def _hilo(i, j, k):
		if k < j:
			j, k = k, j
		if j < i:
			i, j = j, i
		if k < j:
			j, k = k, j
		return i + k

	h = _hilo(r, g, b)
	return rgb2hex(tuple(h - u for u in (r, g, b)))


def fonts_html():
	import matplotlib.font_manager

	def _mk_html(fontname):
		html = "<p>{font}: <span style='font-family:{font}; "
		html += "font-size: 24px;'>{font}  01234 </p>"
		return html.format(font=fontname)

	code = sorted(set([f.name for f in matplotlib.font_manager.fontManager.ttflist]))
	code = [_mk_html(font) for font in code]
	code = "\n".join(code)
	return "<div style='column-count: 2;'>{}</div>".format(code)


def set_style(style: str = 'ticks'):
	sns.set_style(style)
	matplotlib.rcParams['image.interpolation'] = 'none'
	matplotlib.rcParams['grid.linestyle'] = ':'
	return


def create_figure(
		nrows: int = 1,
		ncols: int = 1,
		figsize: Tuple[float, float] = None,
		sharex: str = 'none',
		sharey: str = 'none',
		style: str = 'ticks',
		wspace: float = None,
		hspace: float = None,
		width_ratios: List[float] = None,
		height_ratios: List[float] = None,
		constrained_layout: bool = False,
		tight_layout: bool = False,
		reshape: bool = False,
		dpi: float = None,
		**kwargs, ):
	set_style(style)
	figsize = figsize if figsize else plt.rcParams.get('figure.figsize')
	dpi = dpi if dpi else plt.rcParams.get('figure.dpi')

	fig, axes = plt.subplots(
		nrows=nrows,
		ncols=ncols,
		sharex=sharex,
		sharey=sharey,
		tight_layout=tight_layout,
		constrained_layout=constrained_layout,
		figsize=figsize,
		dpi=dpi,
		gridspec_kw={
			'wspace': wspace,
			'hspace': hspace,
			'width_ratios': width_ratios,
			'height_ratios': height_ratios},
		**kwargs,
	)
	if nrows * ncols > 1 and reshape:
		axes = np.reshape(axes, (nrows, ncols))
	return fig, axes


def save_fig(
		fname: str,
		save_dir: str,
		fig: Union[plt.Figure, List[plt.Figure]],
		sup: Union[plt.Text, List[plt.Text]] = None,
		display: bool = False,
		**kwargs, ):
	defaults = {
		'dpi': 100,
		'bbox_inches': 'tight',
	}
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v
	os.makedirs(save_dir, exist_ok=True)
	save_file = pjoin(save_dir, fname)

	if isinstance(fig, plt.Figure):
		fig.savefig(
			fname=save_file,
			bbox_extra_artists=[sup],
			**kwargs,
		)
	else:
		sup = sup if sup else [None] * len(fig)
		assert fname.split('.')[-1] == 'pdf'
		assert len(fig) == len(sup) > 1
		with PdfPages(save_file) as pages:
			for f, s in zip(fig, sup):
				if f is None:
					continue
				canvas = FigureCanvasPdf(f)
				if s is None:
					canvas.print_figure(
						filename=pages,
						**kwargs,
					)
				else:
					canvas.print_figure(
						filename=pages,
						bbox_extra_artists=[s],
						**kwargs,
					)
	if display:
		if isinstance(fig, list):
			for f in fig:
				plt.show(f)
		else:
			plt.show(fig)
	else:
		if isinstance(fig, list):
			for f in fig:
				plt.close(f)
		else:
			plt.close(fig)
	return save_file
