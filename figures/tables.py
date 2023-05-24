from .fighelper import *


def table_neural(
		df: pd.DataFrame,
		betas: List[float] = None,
		cats: Dict[str, List[str]] = None, ):
	betas = betas if betas else [
		0.5, 0.8, 1, 5]
	cats = cats if cats else {
		'cNVAE': ['fixate1', 'fixate0', 'obj1'],
		'VAE': ['fixate1'],
		'cNAE': ['fixate1'],
		'AE': ['fixate1'],
	}
	table = ''
	for model, cat_list in cats.items():
		first_row_added = False
		for cat in cat_list:
			_df = df.loc[
				(df['model'] == model) &
				(df['category'] == cat)
			]
			if not len(_df):
				continue

			mu = _df.groupby(['beta']).mean()
			sd = _df.groupby(['beta']).std()
			mu, sd = dict(mu['perf']), dict(sd['perf'])

			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				_df = _df.pivot(
					columns='beta',
					values='perf',
					index='index',
				)
			assert len(_df) == 141

			mu['pool'] = _df.max(1).mean()
			sd['pool'] = _df.max(1).std()
			se = {
				k: v / np.sqrt(len(_df))
				for k, v in sd.items()
			}
			if 'VAE' in model:
				r_str = []
				for b in betas:
					if mu[b] > 0.5:
						r_str.append(' \\pm '.join([
							f"$\mathbf{'{'}{mu[b]:0.3f}",
							f"{se[b]:0.3f}{'}'}$",
						]))
					else:
						r_str.append(' \\pm '.join([
							f"${mu[b]:0.3f}",
							f"{se[b]:0.3f}$",
						]))
				r_str = ' & '.join(r_str)
			else:
				if mu['ae'] > 0.5:
					r_str = ' \\pm '.join([
						f"$\mathbf{'{'}{mu['ae']:0.3f}",
						f"{se['ae']:0.3f}{'}'}$",
					])
				else:
					r_str = ' \\pm '.join([
						f"${mu['ae']:0.3f}",
						f"{se['ae']:0.3f}$",
					])

			if first_row_added:
				table += f"& {cat} & {r_str} \\\\\n"
			else:
				table += f"{model} " + f"& {cat} & {r_str} \\\\\n"
				first_row_added = True
	# translate categories to latex
	for k, v in CAT2TEX.items():
		table = table.replace(k, v)
	return table


def table_pvals(
		df: pd.DataFrame,
		betas: List[float] = None, ):
	if betas is None:
		betas = get_betas(df)
	df_pvals = collections.defaultdict(list)
	for b in betas:
		_df = df.loc[
			(df['category'] == 'fixate1') &
			(df['beta'] == b)
			]
		if isinstance(b, float):
			m1, m2 = 'cNVAE', 'VAE'
		else:
			m1, m2 = 'cNAE', 'AE'
		for test in ['perf', 'a']:
			a1 = _df.loc[_df['model'] == m1, test].values
			a2 = _df.loc[_df['model'] == m2, test].values
			good = np.isfinite(a1) & np.isfinite(a2)
			t = sp_stats.ttest_rel(a1[good], a2[good])
			d = (a1 - a2)[good]
			d = d.mean() / d.std(ddof=1)
			df_pvals['m1'].append(m1)
			df_pvals['m2'].append(m2)
			df_pvals['beta'].append(b)
			df_pvals['test'].append(test)
			df_pvals['p'].append(t.pvalue)
			df_pvals['statistic'].append(t.statistic)
			df_pvals['cohens_d'].append(d)
	df_pvals = pd.DataFrame(df_pvals)

	# FDR correction
	from statsmodels.stats.multitest import multipletests
	reject, pvalsc, _, alphac_bonf = multipletests(
		df_pvals['p'].values, method='fdr_bh')
	df_pvals['p_corrected'] = pvalsc
	df_pvals['reject'] = reject

	table = []
	for b in betas:
		_df1 = df_pvals.loc[df_pvals['beta'] == b]
		row = [str(b)]
		for test in ['perf', 'a']:
			_df2 = _df1.loc[_df1['test'] == test]
			row.append(' & '.join([
				f"{_df2['cohens_d'].item():0.2g}",
				f"{_df2['p_corrected'].item():1.1g}",
				'\\cmark' if _df2['reject'].item() else '\\xmark',
			]))
		row = ' & '.join(row)
		table.append(f"{row}\\\\\n")
	table = ''.join(table)

	return table, df_pvals, alphac_bonf
