from .generic import *
from torch import nn
from torch.nn import functional as F
from model.configuration import ConfigVAE


def get_stride_for_cell_type(cell_type: str):
	if cell_type.startswith('normal') \
			or cell_type.startswith('combiner'):
		stride = 1
	elif cell_type.startswith('down'):
		stride = 2
	elif cell_type.startswith('up'):
		stride = -1
	else:
		raise NotImplementedError(cell_type)
	return stride


def print_num_params(module: nn.Module):
	t = PrettyTable(['Module Name', 'Num Params'])
	for name, m in module.named_modules():
		tot = sum(
			p.numel() for p
			in m.parameters()
			if p.requires_grad
		)
		if tot // 1e6 > 0:
			tot = f"{np.round(tot / 1e6, 2):1.1f} M"
		elif tot // 1e3 > 0:
			tot = f"{np.round(tot / 1e3, 2):1.1f} K"

		if '.' not in name:
			if isinstance(m, type(module)):
				t.add_row([m.__class__.__name__, tot])
				t.add_row(['---', '---'])
			else:
				t.add_row([name, tot])
	print(t, '\n\n')
	return


def save_model(
		model: nn.Module,
		chkpt: int = -1, ):
	save_dir = pjoin(
		model.cfg.save_dir,
		model.datetime,
	)
	os.makedirs(save_dir, exist_ok=True)
	fname = type(model).__name__
	fname = '-'.join([
		fname,
		f"{chkpt:04d}",
		f"({now(True)}).bin",
	])
	fname = pjoin(save_dir, fname)
	torch.save(model.state_dict(), fname)
	return fname


def load_model(
		name: str,
		time: int = -1,
		chkpt: int = -1,
		strict: bool = True,
		verbose: bool = False,
		load_dir: str = 'Documents/MTMST/models', ):
	# cfg
	load_dir = pjoin(os.environ['HOME'], load_dir, name)
	fname = next(s for s in os.listdir(load_dir) if 'json' in s)
	with open(pjoin(load_dir, fname), 'r') as f:
		cfg = json.load(f)
	cfg = ConfigVAE(**cfg)
	fname = fname.split('.')[0]
	fname = fname.replace('Config', '')
	if fname == 'VAE':
		from model.vae import VAE
		model = VAE(cfg, verbose=verbose)
	else:
		raise NotImplementedError
	# bin
	load_dir = pjoin(load_dir, 'save')
	time = sorted(os.listdir(load_dir))[time]
	load_dir = pjoin(load_dir, time)
	fname = sorted(os.listdir(load_dir))[chkpt]
	assert fname.split('.')[-1] == 'bin'
	state_dict = pjoin(load_dir, fname)
	state_dict = torch.load(state_dict)
	model.load_state_dict(
		state_dict=state_dict,
		strict=strict,
	)
	model.eval()
	meta = {
		'fname': fname,
		'chkpt': chkpt,
		'dir': load_dir,
	}
	return model, meta
