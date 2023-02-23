from .utils_model import *


class Normal:
	def __init__(
			self,
			mu: torch.Tensor,
			logsigma: torch.Tensor,
			temp: float = 1.0,
			seed: int = None,
	):
		self.mu = soft_clamp(mu)
		logsigma = soft_clamp(logsigma)
		self.sigma = torch.exp(logsigma)  # + 1e-2
		# we don't need above after soft clamp (?)
		if temp != 1.0:
			self.sigma *= temp
		if seed is not None:
			self.rng = torch.Generator()
			self.rng.manual_seed(seed)
		else:
			self.rng = None

	def sample(self):
		return sample_normal(self.mu, self.sigma, self.rng)

	def sample_given_eps(self, eps: torch.Tensor):
		return eps * self.sigma + self.mu

	def log_p(self, samples: torch.Tensor):
		normalized_samples = (samples - self.mu) / self.sigma
		log_p = - 0.5 * (
				normalized_samples ** 2 -
				torch.log(self.sigma) +
				np.log(2 * np.pi)
		)
		return log_p

	def kl(self, normal_dist):
		term1 = (self.mu - normal_dist.mu) / normal_dist.sigma
		term2 = self.sigma / normal_dist.sigma
		return 0.5 * (term1 ** 2 + term2 ** 2) - 0.5 - torch.log(term2)


def soft_clamp(x: torch.Tensor, c: float = 5.0):
	return x.div(c).tanh_().mul(c)


def sample_normal(
		mu: torch.Tensor,
		sigma: torch.Tensor,
		generator: torch.Generator = None, ):
	eps = mu.mul(0).normal_(generator=generator)
	z = eps.mul_(sigma).add_(mu)
	return z, eps


@torch.jit.script
def soft_clamp_jit(x: torch.Tensor, c: float = 5.0):
	return x.div(c).tanh_().mul(c)
	# 5. * torch.tanh(x / 5.) <--> soft differentiable clamp between [-5, 5]


@torch.jit.script
def sample_normal_jit(
		mu: torch.Tensor,
		sigma: torch.Tensor, ):
	eps = mu.mul(0).normal_()
	z = eps.mul_(sigma).add_(mu)
	return z, eps
