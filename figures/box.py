from .fighelper import *


def plot_box(
		df: pd.DataFrame,
		y: str,
		ylabel: str = '',
		xlabel: str = r'$\beta$',
		cat: str = 'fixate1',
		display: bool = True,
		**kwargs, ):
	defaults = dict(
		pal=get_palette()[0],
		labelsize_x=13,
		labelsize_y=11,
		tick_labelsize_x=11,
		tick_labelsize_y=10,
		bbox=(1.0, 1.06),
		figsize=(10, 2),
		ylim=(0, 1),
	)
	kwargs = setup_kwargs(defaults, kwargs)
	betas = get_betas(df)

	_df = df.loc[
		(df['category'] == cat) &
		(df['beta'] != 'ae')
		]
	_df_ae = df.loc[
		(df['category'] == cat) &
		(df['beta'] == 'ae')
		]
	_pal = {
		k: v for k, v in kwargs['pal'].items()
		if k in _df['model'].unique()
	}
	_pal_ae = {
		k: v for k, v in kwargs['pal'].items()
		if k in _df_ae['model'].unique()
	}
	fig, ax = create_figure(1, 1, kwargs['figsize'])
	sns.boxplot(
		data=_df,
		x='beta',
		y=y,
		hue='model',
		order=betas,
		palette=_pal,
		hue_order=_pal,
		**_PROPS,
		ax=ax,
	)
	sns.boxplot(
		data=_df_ae,
		x='beta',
		y=y,
		hue='model',
		order=betas,
		palette=_pal_ae,
		hue_order=_pal_ae,
		**_PROPS,
		ax=ax,
	)
	ax.set_xlabel(
		xlabel=xlabel,
		fontsize=kwargs['labelsize_x'],
	)
	ax.set_ylabel(
		ylabel=ylabel,
		fontsize=kwargs['labelsize_y'],
	)
	ax.tick_params(
		axis='x',
		labelsize=kwargs['tick_labelsize_x'],
	)
	ax.tick_params(
		axis='y',
		labelsize=kwargs['tick_labelsize_y'],
	)
	ax.set_ylim(kwargs['ylim'])
	ax.grid()

	leg = ax.get_legend()
	if leg is not None:
		leg.set_bbox_to_anchor(kwargs['bbox'])

	if display:
		plt.show()
	else:
		plt.close()
	return fig, ax


_PROPS = {
	'meanprops': {
		'marker': 'o',
		'markerfacecolor': 'white',
		'markeredgecolor': 'k',
		'markersize': 3,
		'alpha': 1.0},
	'boxprops': {
		'edgecolor': 'k',
		# 'facecolor': 'none',
		# 'lw': 1.5,
		'ls': '-'},
	'medianprops': {'color': 'k', 'lw': 1.5},
	'whiskerprops': {'color': 'k', 'lw': 1.5},
	'capprops': {'color': 'k', 'lw': 1.5, 'zorder': 3},
	'flierprops': {
		'marker': 'o',
		'markersize': 2,
		'alpha': 0.3,
		'zorder': 1},
	'showfliers': True,
	'showmeans': True,
	'dodge': True,
	'width': 0.5,
}
