from .generic import *
import scipy.io as sio
DESIGNSZ = 30


def process_crcns(g: h5py.Group, path: str):
	translate = {
		'rfloc': 'rf_loc',
		'spkbinned': 'spks',
		'opticflows': 'hf_params',
		'aperturecenter': 'hf_center',
	}
	data_all = {}
	for f in sorted(os.listdir(path)):
		if not f.endswith('.mat'):
			continue
		data = pjoin(path, f)
		data = sio.loadmat(data)['mtdata']
		fields = list(data.dtype.fields)
		data = data.item()

		processed = {}
		for k, v in zip(fields, data):
			key = k.lower()
			if key in translate:
				key = translate[key]
			if np.prod(v.size) == 1:
				processed[key] = v.item()
			else:
				processed[key] = v
		key = '_'.join([
			processed['cellid'],
			f.split('.')[0],
		])
		data_all[key] = processed
	crcns_all = []
	for key, processed in sorted(data_all.items()):
		k = key.split('_')
		if len(k) == 3:
			cellindx = int(k[1])
		else:
			cellindx = 1
		k_final = '_'.join([k[0], k[-1]])
		group = g.create_group(k_final)
		crcns_all.append(k_final)
		# main
		spks = processed['spks'].astype(float)
		eyeloc = processed['eyeloc'].astype(float)
		hf_params = processed['hf_params'].astype(float)
		hf_center = processed['hf_center'].astype(float)
		hf_radius = processed['aperturediameter'] / 2
		hf_radius = hf_radius * np.ones(len(hf_center))
		# create dataset
		data = [
			('spks', float, spks),
			('eyeloc', float, eyeloc),
			('hf_params', float, hf_params),
			('hf_center', float, hf_center),
			('hf_radius', float, hf_radius),
			('hf_radius_ratio', float, hf_radius / DESIGNSZ),
		]
		for a, b, c in data:
			group.create_dataset(a, dtype=b, data=c)

		# attrs
		attrs = {
			'expt_name': k[0],
			'cellindex': cellindx,
			'n_channels': spks.shape[1],
			'rf_loc': processed['rf_loc'],
			'dt': processed['dt'],
			'designsize': DESIGNSZ,
			'diameter': processed['aperturediameter'],
			'has_repeats': False,
		}
		group.attrs.update(attrs)
	return crcns_all


def process_mtmst(g: h5py.Group, path: str, tres: int):
	mat_files = [
		f for f in os.listdir(path)
		if f"tres{tres}" in f
	]
	expt_all = {}
	for f in sorted(mat_files):
		mat_content = sio.loadmat(pjoin(path, f))
		expt_name = mat_content['expt_name'].item()
		group = g.create_group(expt_name)

		# main
		lfp = mat_content['lfp'].astype(float)
		spks = mat_content['spks'].astype(float)
		badspks = mat_content['badspks'].astype(bool)
		hf_params = mat_content['opticflows'].astype(float)
		hf_center = np.concatenate([
			mat_content['centerx'],
			mat_content['centery'],
		], axis=-1).astype(float)
		diameter = mat_content['diameter'][0].astype(float)
		partition = mat_content['partition'][0].astype(int)
		hf_radius = np.zeros(len(hf_center))
		for i in range(len(partition) - 1):
			intvl = range(partition[i], partition[i + 1])
			hf_radius[intvl] = diameter[i] / 2
		assert not any(hf_radius == 0)

		# create datasets
		data = [
			('lfp', float, lfp),
			('spks', float, spks),
			('badspks', bool, badspks),
			('hf_params', float, hf_params),
			('hf_center', float, hf_center),
			('hf_radius', float, hf_radius),
			('hf_radius_ratio', float, hf_radius / DESIGNSZ),
		]
		for a, b, c in data:
			group.create_dataset(a, dtype=b, data=c)

		# attrs
		attrs = {
			'expt_name': expt_name,
			'cellindex': mat_content['cellindex'].item(),
			'n_channels': spks.shape[1],
			'dt': tres,
			'designsize': DESIGNSZ,
			'nx': mat_content['nx'].item(),
			'ny': mat_content['ny'].item(),
			'field': mat_content['field'],
			'rf_loc': mat_content['rf_loc'],
			'spatres': mat_content['spatres'],
			'latency': mat_content['latency'].item(),
			'diameter': diameter,
			'partition': partition,
			'has_repeats': mat_content['repeats'].squeeze().astype(int).size > 0,
		}
		if attrs['has_repeats']:
			attrs.update({
				'diameterR': mat_content['diameterR'][0].astype(float),
				'partitionR': mat_content['partitionR'][0].astype(int),
			})
		expt_all[expt_name] = attrs['n_channels']
		group.attrs.update(attrs)

		# repeats?
		if attrs['has_repeats']:
			lfp_r = mat_content['lfpR'].astype(float)
			spks_r = mat_content['spksR'].astype(float)
			badspks_r = mat_content['badspksR'].astype(bool)
			hf_params_r = mat_content['opticflowsR'].astype(float)
			hf_center_r = np.concatenate([
				mat_content['centerxR'],
				mat_content['centeryR'],
			], axis=-1).astype(float)
			diameter_r = mat_content['diameterR'][0].astype(float)
			partition_r = mat_content['partitionR'][0].astype(int)
			hf_radius_r = np.zeros(len(hf_center_r))
			for i in range(len(partition_r) - 1):
				intvl = range(partition_r[i], partition_r[i + 1])
				hf_radius_r[intvl] = diameter_r[i] / 2
			assert not any(hf_radius_r == 0)
			psth_raw_all = mat_content['psth_raw_all'].astype(int)
			fix_lost_all = mat_content['fix_lost_all'].astype(bool)
			tind_start_all = mat_content['tind_start_all'].astype(int)

			assert spks.shape[1] == spks_r.shape[1] == \
				len(psth_raw_all) == len(tind_start_all) == len(fix_lost_all)

			# create datasets
			data = [
				('lfpR', float, lfp_r),
				('spksR', float, spks_r),
				('badspksR', bool, badspks_r),
				('hf_paramsR', float, hf_params_r),
				('hf_centerR', float, hf_center_r),
				('hf_radiusR', float, hf_radius_r),
				('hf_radius_ratioR', float, hf_radius_r / DESIGNSZ),
				('fix_lost_all', int, fix_lost_all),
				('psth_raw_all', float, psth_raw_all),
				('tind_start_all', int, tind_start_all),
			]
			for a, b, c in data:
				group.create_dataset(a, dtype=b, data=c)

	return expt_all


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


def _setup_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--tres",
		help='temporal resolution',
		type=int,
		default=25,
	)
	parser.add_argument(
		"--save_dir",
		help='path to save data',
		type=str,
		default='/home/hadi/Documents/MTMST/data',
	)
	return parser.parse_args()


def _main():
	args = _setup_args()
	print(args)

	# create h5 file
	file = f"ALL_tres{args.tres}.h5"
	file = pjoin(args.save_dir, file)
	file = h5py.File(file, 'w')

	base_dir = '/home/hadi/Documents/MTMST-other'

	# CRCNS
	g = file.create_group('CRCNS')
	path = 'CRCNS/data'
	path = pjoin(base_dir, path)
	crcns_all = process_crcns(g, path)
	file.attrs.update({
		'CRCNS_expts': crcns_all,
		'CRCNS_nch': [1] * len(crcns_all),
	})
	print('[PROGRESS] CRCNS done.')
	# YUWEI
	g = file.create_group('YUWEI')
	path = 'Yuwei/MTproject_data/xtracted'
	path = pjoin(base_dir, path)
	yuwei_all = process_mtmst(
		g, path, tres=args.tres)
	file.attrs.update({
		'YUWEI_expts': list(yuwei_all.keys()),
		'YUWEI_nch': list(yuwei_all.values()),
	})
	print('[PROGRESS] YUWEI done.')
	# NARDIN
	g = file.create_group('NARDIN')
	path = 'Nardin/MTproject_data/xtracted'
	path = pjoin(base_dir, path)
	nardin_all = process_mtmst(
		g, path, tres=args.tres)
	file.attrs.update({
		'NARDIN_expts': list(nardin_all.keys()),
		'NARDIN_nch': list(nardin_all.values()),
	})
	print('[PROGRESS] NARDIN done.')
	# close file
	file.close()

	print(f"\n[PROGRESS] processing ephys data done ({now(True)}).\n")
	return


if __name__ == "__main__":
	_main()
