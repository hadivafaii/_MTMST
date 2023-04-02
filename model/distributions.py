from .utils_model import *


class Normal:
	def __init__(
			self,
			mu: torch.Tensor,
			logsig: torch.Tensor,
			temp: float = 1.0,
			seed: int = None,
			device: torch.device = None,
	):
		self.mu = mu
		logsig = softclamp(logsig, 5)
		self.sigma = torch.exp(logsig)
		if temp != 1.0:
			assert temp >= 0
			self.sigma *= temp
		if seed is not None:
			self.rng = torch.Generator(device)
			self.rng.manual_seed(seed)
		else:
			self.rng = None

	def sample(self):
		if self.rng is None:
			return sample_normal_jit(self.mu, self.sigma)
		else:
			return sample_normal(self.mu, self.sigma, self.rng)

	def log_p(self, samples: torch.Tensor):
		zscored = (samples - self.mu) / self.sigma
		log_p = (
			- 0.5 * zscored.pow(2)
			- 0.5 * np.log(2*np.pi)
			- torch.log(self.sigma)
		)
		return log_p

	def kl(self, p):
		term1 = (self.mu - p.mu) / p.sigma
		term2 = self.sigma / p.sigma
		kl = 0.5 * (
			term1.pow(2) + term2.pow(2) +
			torch.log(term2).mul(-2) - 1
		)
		return kl


@torch.jit.script
def softclamp(x: torch.Tensor, c: float):
	return x.div(c).tanh_().mul(c)


@torch.jit.script
def sample_normal_jit(
		mu: torch.Tensor,
		sigma: torch.Tensor, ):
	eps = torch.empty_like(mu).normal_()
	return sigma * eps + mu


def sample_normal(
		mu: torch.Tensor,
		sigma: torch.Tensor,
		rng: torch.Generator = None, ):
	eps = torch.empty_like(mu).normal_(
		mean=0., std=1., generator=rng)
	return sigma * eps + mu


@torch.jit.script
def residual_kl(
		delta_mu: torch.Tensor,
		delta_sig: torch.Tensor,
		sigma: torch.Tensor, ):
	return 0.5 * (
		delta_sig.pow(2) - 1 +
		(delta_mu / sigma).pow(2)
	) - torch.log(delta_sig)


def explin(p: torch.Tensor):
	above = torch.clamp(p, min=0)
	below = torch.clamp(p, max=0)
	return torch.where(
		condition=p <= 0,
		input=below.exp(),
		other=above + 1,
	)
