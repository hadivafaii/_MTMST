from utils.generic import *
from vae.vae2d import VAE


def traverse(
		vae: VAE,
		group_i: int,
		latent_i: int,
		z: List[torch.Tensor] = None,
		z_interp: torch.Tensor = None,
		mu: float = 0.0,
		sd: float = 1.0,
		n_sd: int = 3,
		steps: int = 16, ):
	if z is None:
		_, z, _ = vae.sample()

	if z_interp is None:
		z_interp = torch.linspace(
			mu - n_sd * sd,
			mu + n_sd * sd,
			steps=steps,
		).unsqueeze(-1).unsqueeze(-1)
	z_interp = z_interp.to(z[0].device)

	d = vae.cfg.input_sz
	x_gen = (len(z[0]), steps, 2, d, d)
	x_gen = np.empty(x_gen)
	for sample_i in range(len(x_gen)):
		z_trav = z.copy()
		z_trav = [
			torch.repeat_interleave(
				e[[sample_i]],
				repeats=steps,
				dim=0,
			) for e in z_trav
		]
		z_trav[group_i][:, latent_i] = z_interp
		x_gen[sample_i] = to_np(vae.generate(z_trav))
	return x_gen
