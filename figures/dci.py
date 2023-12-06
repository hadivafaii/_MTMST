from .fighelper import *


def plot_bar_dci(
		df: pd.DataFrame,
		cat: str = 'fixate1',
		annotate: bool = True,
		display: bool = True,
		**kwargs, ):
	defaults = dict(
		pal=get_palette()[0],
		labelsize_x=13,
		xlabel=r'$\beta$',
		tick_labelsize_x=9,
		tick_labelsize_y=10,
		figsize=(3.9, 3.3),
		bbox=(1, 1.07),
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
	dci = {
		'i': 'Informativeness',
		'd': 'Disentanglement',
		'c': 'Completeness',
	}
	xy_frac = {
		'i': (0.576, 0.818),
		'd': (0.548, 0.815),
		'c': (0.622, 0.815),
	}
	fig, axes = create_figure(
		nrows=3,
		ncols=1,
		figsize=kwargs['figsize'],
		sharex='all',
		sharey='all',
		layout='constrained',
	)
	for i, (metric, label) in enumerate(dci.items()):
		sns.barplot(
			data=_df,
			x='beta',
			y=metric,
			hue='model',
			order=betas,
			dodge=True,
			width=0.7,
			hue_order=_pal,
			palette=_pal,
			ax=axes[i],
		)
		sns.barplot(
			data=_df_ae,
			x='beta',
			y=metric,
			hue='model',
			order=betas,
			dodge=True,
			width=0.7,
			hue_order=_pal_ae,
			palette=_pal_ae,
			ax=axes[i],
		)
		axes[i].set(xlabel='', ylabel='')
		if annotate:
			axes[i].annotate(
				text=label,
				xy=xy_frac[metric],
				xycoords='axes fraction',
				fontsize=10,
			)
		leg = axes[i].get_legend()
		if leg is not None and i > 0:
			leg.remove()
	add_grid(axes)
	for ax in axes.flat:
		ax.tick_params(
			axis='y',
			labelsize=kwargs['tick_labelsize_y'],
		)

	axes[-1].tick_params(
		axis='x',
		rotation=-90,
		labelsize=kwargs['tick_labelsize_x'],
	)
	axes[-1].set_xlabel(
		xlabel=kwargs['xlabel'],
		fontsize=kwargs['labelsize_x'],
	)
	axes[-1].set_ylim((0, 0.92))

	if kwargs['bbox'] is not None:
		sns.move_legend(
			axes[0],
			loc='best',
			title='',
			fontsize=6.5,
			bbox_to_anchor=kwargs['bbox'],
		)
	else:
		axes[0].get_legend().remove()

	if display:
		plt.show()
	else:
		plt.close()
	return fig, axes


def plot_bar_untangle(
		df: pd.DataFrame,
		pal: Dict[str, str],
		display: bool = True, ):
	fig, ax = create_figure(1, 1, (10, 3))
	bp = sns.barplot(
		data=df,
		x='f',
		y='r2',
		hue='model',
		hue_order=['cNVAE', 'VAE', 'PCA', 'cNAE', 'AE'],
		palette=pal,
		dodge=True,
		width=0.795,
		ax=ax,
	)
	barplot_add_vals(
		bp,
		frac_x=0.41,
		frac_y=0.10,
		fontsize=8.0,
		rotation=-90,
		ha='center',
		color='w',
		decimals=2,
	)
	ax.set_ylim(bottom=-0.03)
	ax.set_ylabel(r'$R^2$', fontsize=13)
	ax.tick_params(axis='y', labelsize=10)
	ax.set_xticklabels(LBL2TEX.values())
	ax.set_xlabel('')
	ax.grid()

	leg = ax.get_legend()
	leg.set_bbox_to_anchor((1, 1.04))

	if display:
		plt.show()
	else:
		plt.close()
	return fig, ax


def plot_scatter(
		data: dict,
		display: bool = True,
		**kwargs, ):
	defaults = dict(
		pal=get_palette()[0],
		labelsize_x=13,
		labelsize_y=11,
		tick_labelsize_x=11,
		tick_labelsize_y=10,
		bbox=(1.0, 1.06),
		figsize=(9.5, 1.9),
		ylim=(0, 1),
	)
	kwargs = setup_kwargs(defaults, kwargs)
	g, _, select_lbl = prep_rofl()
	fig, axes = create_figure(
		nrows=2,
		ncols=g['vld'].shape[1],
		figsize=kwargs['figsize'],
		sharex='col',
		sharey='row',
		layout='constrained',
		style='white',
	)
	for i, (fit, v) in enumerate(data.items()):
		model = extract_info(fit)[-1]
		y = v['data_vld']['z']
		r = 1 - sp_dist.cdist(
			XA=y.T,
			XB=g['vld'].T,
			metric='correlation',
		).T
		inds = np.argmax(np.abs(r), axis=1)
		for j in range(g['vld'].shape[1]):
			_x = g['vld'][:, j]
			_y = y[:, inds[j]]
			_y = sp_stats.zscore(_y)

			# lin regress
			lr = sp_stats.linregress(_x, _y)
			x_min = _x.min()
			x_max = _x.max()
			xs = np.linspace(x_min, x_max, 1000)
			ys = lr.slope * xs + lr.intercept

			sns.scatterplot(
				x=_x,
				y=_y,
				s=5,
				alpha=0.05,
				# bins=20,
				color=kwargs['pal'][model],
				ax=axes[i, j],
			)
			axes[i, j].plot(xs, ys, color='white', ls='--', lw=1.0, alpha=0.7)

			axes[i, j].annotate(
				text=r'$r = $' + str(np.round(r[j, inds[j]], 2)),
				xy=(0.05 if r[j, inds[j]] > 0 else 0.42, 0.865),
				xycoords='axes fraction',
				fontsize=7,
			)
			if j == 0:
				axes[i, j].set_ylabel(model)
			if i == 0:
				axes[i, j].set_title(
					LBL2TEX[select_lbl[j]],
					y=1.02,
				)
	axes[0, 0].set_ylim(-3, 3)
	axes[1, 0].set_ylim(-3, 3)
	remove_ticks(axes, False)
	ax_square(axes)
	if display:
		plt.show()
	else:
		plt.close()
	return fig, axes
