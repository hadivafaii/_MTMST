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
		# return sample_normal_jit(self.mu, self.sigma)

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
			term1.pow(2) + term2.pow(2)
			- 2 * torch.log(term2) - 1
		)
		return kl


def soft_clamp(x: torch.Tensor, c: float = 5.0):
	return x.div(c).tanh_().mul(c)


def sample_normal(
		mu: torch.Tensor,
		sigma: torch.Tensor,
		generator: torch.Generator = None, ):
	eps = mu.mul(0).normal_(
		generator=generator).mul_(sigma)
	z = eps.add(mu)
	return z


# @torch.jit.script
# def soft_clamp_jit(x):
	# return x.div(5).tanh_().mul(5)
	# 5. * torch.tanh(x / 5.) <--> soft differentiable clamp between [-5, 5]


# @torch.jit.script
# def sample_normal_jit(mu, sigma):
	# eps = mu.mul(0).normal_()
	# z = eps.mul_(sigma).add(mu)
	# return z
