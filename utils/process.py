from .generic import *
import scipy.io as sio


def load_cellinfo(load_dir: str):
	clu = pd.read_csv(pjoin(load_dir, "cellinfo.csv"))
	ytu = pd.read_csv(pjoin(load_dir, "cellinfo_ytu.csv"))
	clu = clu[np.logical_and(1 - clu.SingleElectrode, clu.HyperFlow)]
	ytu = ytu[np.logical_and(1 - ytu.SingleElectrode, ytu.HyperFlow)]

	useful_cells = {}
	for name in clu.CellName:
		useful_ch = []
		for i in range(1, 16 + 1):
			if clu[clu.CellName == name][f"chan{i}"].item():
				useful_ch.append(i - 1)
		if len(useful_ch) > 1:
			useful_cells[name] = useful_ch
	for name in ytu.CellName:
		useful_ch = []
		for i in range(1, 24 + 1):
			if ytu[ytu.CellName == name][f"chan{i}"].item():
				useful_ch.append(i - 1)
		if len(useful_ch) > 1:
			useful_cells[name] = useful_ch

	return useful_cells


# TODO: fix issue with spkst
def mat2h5py(
		load_dir: str,
		save_dir: str,
		file_name: str,
		tres: int = 25,
		grd: int = 15, ):
	file_name = f"{file_name}_tres{tres:d}.h5"
	file_name = pjoin(save_dir, file_name)
	ff = h5py.File(file_name, 'w')
	mat_files = sorted(os.listdir(load_dir))
	mat_files = [f for f in mat_files if f"tres{tres}" in f]
	pbar = tqdm(mat_files)
	for f in pbar:
		mat_content = sio.loadmat(pjoin(load_dir, f))

		expt_name = mat_content['expt_name'].item()
		group = ff.create_group(expt_name)
		msg = f'group {expt_name} created'
		pbar.set_description(msg)

		# main
		lfp = mat_content['lfp'].astype(float)
		spks = mat_content['spks'].astype(float)
		# spkst = _fix_spkst(mat_content['spkst']).astype(float)
		badspks = mat_content['badspks'].astype(bool)
		fixlost = mat_content['fixlost'].astype(bool)
		partition = mat_content['partition'][0].astype(int)
		hyperflow = np.concatenate([
			mat_content['centerx'],
			mat_content['centery'],
			mat_content['opticflows']
		], axis=-1).astype(float)
		stim1 = mat_content['stim1'].astype(float)
		stim2 = mat_content['stim2'].astype(float)

		# metadata
		rf_loc = mat_content['rf_loc'].squeeze().astype(float)
		field = mat_content['field'].squeeze().astype(float)
		cellindex = mat_content['cellindex'].item()
		latency = mat_content['latency'].item()
		spatres = mat_content['spatres'].squeeze().astype(float)
		nx = mat_content['nx'].item()
		ny = mat_content['ny'].item()
		num_ch = spks.shape[1]

		# create datasets
		group.create_dataset('lfp', dtype=float, data=lfp)
		group.create_dataset('spks', dtype=float, data=spks)
		# group.create_dataset('spkst', dtype=float, data=spkst)
		group.create_dataset('badspks', dtype=bool, data=badspks)
		group.create_dataset('fixlost', dtype=bool, data=fixlost)
		group.create_dataset('partition', dtype=int, data=partition)
		group.create_dataset('hyperflow', dtype=float, data=hyperflow)
		group.create_dataset('stim1', dtype=float, data=_fix_stim(stim1, grd))
		group.create_dataset('stim2', dtype=float, data=_fix_stim(stim2, grd))
		group.create_dataset('rf_loc', dtype=float, data=rf_loc)
		group.create_dataset('field', dtype=float, data=field)
		group.create_dataset('cellindex', dtype=int, data=cellindex)
		group.create_dataset('latency', dtype=int, data=latency)
		group.create_dataset('spatres', dtype=float, data=spatres)
		group.create_dataset('nx', dtype=int, data=nx)
		group.create_dataset('ny', dtype=int, data=ny)

		# repeats?
		repeats = mat_content['repeats'].squeeze().astype(int)
		if repeats.size:
			lfp_r = mat_content['lfpR'].astype(float)
			spks_r = mat_content['spksR'].astype(float)
			# spkst_r = _fix_spkst(mat_content['spkstR']).astype(float)
			badspks_r = mat_content['badspksR'].astype(bool)
			fixlost_r = mat_content['fixlostR'].astype(bool)
			partition_r = mat_content['partitionR'][0].astype(int)
			hyperflow_r = np.concatenate([
				mat_content['centerxR'],
				mat_content['centeryR'],
				mat_content['opticflowsR']
			], axis=-1).astype(float)
			stim_r = mat_content['stimR'].astype(float)
			psth_raw_all = mat_content['psth_raw_all'].astype(int)
			fix_lost_all = mat_content['fix_lost_all'].astype(bool)
			tind_start_all = mat_content['tind_start_all'].astype(int)

			assert num_ch == spks_r.shape[1] == len(psth_raw_all) \
				== len(fix_lost_all) == len(tind_start_all)

			# create datasets
			subgroup = group.create_group('repeats')
			subgroup.create_dataset('lfpR', dtype=float, data=lfp_r)
			subgroup.create_dataset('spksR', dtype=float, data=spks_r)
			# subgroup.create_dataset('spkstR', dtype=float, data=spkst_r)
			subgroup.create_dataset('badspksR', dtype=bool, data=badspks_r)
			subgroup.create_dataset('fixlostR', dtype=bool, data=fixlost_r)
			subgroup.create_dataset('partitionR', dtype=int, data=partition_r)
			subgroup.create_dataset('hyperflowR', dtype=float, data=hyperflow_r)
			subgroup.create_dataset('stimR', dtype=float, data=_fix_stim(stim_r, grd))
			subgroup.create_dataset('psth_raw_all', dtype=float, data=psth_raw_all)
			subgroup.create_dataset('fix_lost_all', dtype=int, data=fix_lost_all)
			subgroup.create_dataset('tind_start_all', dtype=int, data=tind_start_all)

	print('\nDONE.')
	ff.close()
	return


def _fix_stim(x, grd):
	return np.swapaxes(np.moveaxis(x.reshape(
		(-1, 2, grd, grd)), 1, -1), 1, 2)


def _fix_spkst(x):
	num_ch = len(x)
	longest = 0
	for a in x:
		longest = max(longest, len(a.item()))
	y = np_nans((longest, num_ch))
	for i, a in enumerate(x):
		data = a.item().squeeze()
		if data.size and data.shape:
			y[:len(data), i] = data
	return y
