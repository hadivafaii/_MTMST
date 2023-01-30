from .helper import *
from scipy.spatial.transform import Rotation


class OpticFlow(Obj):
	def __init__(
			self,
			fov: float = 45,
			res: float = 0.1,
			obj_r: float = 0.2,
			z_bg: float = 1,
			seed: int = 0,
			**kwargs,
	):
		super(OpticFlow, self).__init__(**kwargs)
		assert z_bg > 0
		self.fov = fov
		self.res = res
		self.z_bg = z_bg
		self.obj_r = obj_r
		self.rng = get_rng(seed)
		self._compute_span()
		self._compute_polar_coords()

	def _set_params(self, vel, obj_pos, obj_vel):
		self.obj_pos, self.obj_vel = _check_obj(
			obj_pos, obj_vel)
		self.vel = _check_input(vel, -1)
		return

	def compute_flow(
			self,
			vel: np.ndarray,
			obj_pos: np.ndarray,
			obj_vel: np.ndarray, ):
		self._set_params(vel, obj_pos, obj_vel)
		# add object
		v_transl_obj, x_obj = self._add_obj()
		x = _expand(self.x, self.obj_pos.shape[1], -1)
		x[~np.isnan(x_obj)] = x_obj[~np.isnan(x_obj)]
		# apply self movement
		v_rot = self._compute_v_rot(x=x)
		v_transl = self._compute_v_tr()
		v_transl = _expand(v_transl, x.shape[-1], -1)
		# expand/merge together
		kws = {
			'reps': self.vel.shape[1],
			'axis': -2,
		}
		x = _expand(x, **kws)
		nans = _expand(np.isnan(x_obj), **kws)
		v_transl_obj = _expand(v_transl_obj, **kws)
		v_transl[~nans] = v_transl_obj[~nans]
		# compute retinal velocity
		alpha_dot = self._compute_alpha_dot(
			v=v_transl - v_rot, x=x, axis=3)
		return v_transl_obj, x, v_rot, v_transl, alpha_dot

	def compute_coords(self, fix: np.ndarray = (0, 0)):
		self._compute_fix(fix)
		self._compute_rot()
		self._compute_xyz()
		return self

	def _add_obj(
			self,
			pos: np.ndarray = None,
			vel: np.ndarray = None, ):
		if pos is None:
			pos = self.obj_pos
		if vel is None:
			vel = self.obj_vel
		v_transl = self._compute_v_tr(vel)

		shape = (1,) * self.x.ndim
		shape += (pos.shape[1],)
		shape = np.array(shape)
		# find obj X (reality)
		x = np.expand_dims(self.x, -1)
		x = np.repeat(x, shape[-1], -1)
		x *= pos[-1].reshape(shape)
		x_real = 'aij, amnjc -> amnic'
		x_real = np.einsum(x_real, self.R, x)
		# apply mask
		shape[-2] = 2
		mask = x_real[..., :2, :] - pos[:2].reshape(shape)
		mask = sp_lin.norm(mask, axis=-2) < self.obj_r
		mask = _expand(mask, 3, -2)
		v_transl[~mask] = 0
		x[~mask] = np.nan
		return v_transl, x

	def _compute_alpha_dot(
			self,
			v: np.ndarray,
			x: np.ndarray = None,
			axis: int = 3, ):
		if x is None:
			x = self.x
		delta = v.ndim - x.ndim
		if delta > 0:
			for n in v.shape[-delta:]:
				x = _expand(x, n, -1)
		alpha_dot = []
		for i in [0, 1]:
			a = (
				v.take(i, axis) * x.take(2, axis) -
				v.take(2, axis) * x.take(i, axis)
			)
			a /= sp_lin.norm(
				x.take([i, 2], axis),
				axis=axis,
			) ** 2
			a = np.expand_dims(a, axis)
			alpha_dot.append(a)
		alpha_dot = np.concatenate(alpha_dot, axis)
		return alpha_dot

	def _compute_v_tr(self, v: np.ndarray = None):
		if v is None:
			v = -self.vel
		v_tr = np.dot(np.transpose(self.R, (0, 2, 1)), v)
		v_tr = np.repeat(np.repeat(np.expand_dims(np.expand_dims(
			v_tr, 1), 1), self.dim, 1), self.dim, 2)
		return v_tr

	def _compute_v_rot(
			self,
			v: np.ndarray = None,
			x: np.ndarray = None, ):
		if v is None:
			v = -self.vel
		if x is None:
			x = self.x
		fix_norm = sp_lin.norm(
			self.fix,
			axis=-1,
			keepdims=True,
		) ** 2
		# Normal velocity
		mag = (self.fix @ v) / fix_norm
		v_normal = 'ab, ai -> aib'
		v_normal = np.einsum(v_normal, mag, self.fix)
		v_normal = np.expand_dims(v, 0) - v_normal
		# Rotational velocity
		omega = 'aij, ajb -> aib'
		omega = np.einsum(omega, skew(self.fix, 1), v_normal)
		omega /= np.expand_dims(fix_norm, axis=-1)
		omega = skew(omega, 1)
		mat = 'ani, anmb, amj -> aijb'
		mat = np.einsum(mat, self.R, omega, self.R)
		if x.ndim - mat.ndim == 0:
			v_rot = 'aijb, axyj -> axyib'
		elif x.ndim - mat.ndim == 1:
			v_rot = 'aijb, axyjc -> axyibc'
		else:
			raise RuntimeError
		v_rot = np.einsum(v_rot, mat, x)
		return v_rot

	def _compute_xyz(self):
		gamma = 'aij, mnj -> amni'
		gamma = np.einsum(gamma, self.R, self.tan)
		shape = (1, self.dim, self.dim, 1)
		self.x = self.z_bg * np.concatenate([
			self.tan[..., 0].reshape(shape),
			self.tan[..., 1].reshape(shape),
			np.ones(shape),
		], axis=-1) / gamma[..., [-1]]
		return

	def _compute_rot(self):
		r = cart2polar(self.fix)
		u0 = np.concatenate([
			- np.sin(r[:, [2]]),
			np.cos(r[:, [2]]),
			np.zeros((len(r), 1)),
		], axis=-1)
		u0 = np.array(u0)
		self.R = Rotation.from_rotvec(
			r[:, [1]] * u0).as_matrix()
		return

	def _compute_fix(self, fix: np.ndarray = None):
		upper = 1 / np.tan(np.deg2rad(self.fov))
		if fix is None:
			x0 = self.rng.uniform(
				low=-upper,
				high=upper,
			)
			y0 = self.rng.uniform(
				low=-upper + abs(x0),
				high=upper - abs(x0),
			)
			fix = (x0, y0)
		fix = _check_input(fix, 0)
		assert fix.shape[1] == 2, "fix = (X0, Y0)"

		passed = np.abs(fix).sum(1) < upper
		fix_z = self.z_bg * np.ones((passed.sum(), 1))
		self.fix = np.concatenate([
			fix[passed],
			fix_z,
		], axis=-1)
		return

	def _compute_polar_coords(self):
		a, b = np.meshgrid(self.span, self.span)
		self.alpha = np.concatenate([
			np.expand_dims(a, -1),
			np.expand_dims(b, -1),
		], axis=-1)
		self.tan = np.concatenate([
			np.tan(self.alpha),
			np.ones((self.dim,) * 2 + (1,)),
		], axis=-1)
		return

	def _compute_span(self):
		dim = self.fov / self.res
		assert dim.is_integer()
		self.dim = 2 * int(dim) + 1
		self.span = np.deg2rad(np.linspace(
			-self.fov, self.fov, self.dim))
		return


class HyperFlow(Obj):
	def __init__(
			self,
			opticflow: np.ndarray,
			center: np.ndarray,
			size: int = 32,
			sres: float = 1,
			radius: float = 8,
			**kwargs,
	):
		super(HyperFlow, self).__init__(**kwargs)
		assert len(opticflow) == len(center)
		assert opticflow.shape[1] == 6
		assert center.shape[1] == 2
		self.opticflow = opticflow
		self.center = center
		self.size = size
		self.sres = sres
		self.radius = radius
		self.stim = None

	def compute_hyperflow(self):
		self.stim = self._compute_hyperflow().reshape(
			(-1,) + (self.size//self.sres,) * 2 + (2,))
		return self.stim

	def show_psd(
			self,
			attr: str = 'opticflow',
			log: bool = True,
			**kwargs, ):
		defaults = {
			'fig_x': 2.2,
			'fig_y': 4.5,
			'lw': 0.8,
			'tight_layout': True,
			'ylim_bottom': 1e-5 if log else 0,
			'c': 'C0' if attr == 'opticflow' else 'k',
			'cutoff': 2 if attr == 'opticflow' else 0.2,
			'fs': 1000 / self.tres,
			'detrend': False,
		}
		for k, v in defaults.items():
			if k not in kwargs:
				kwargs[k] = v
		px = getattr(self, attr).T
		f, px = sp_sig.periodogram(
			x=px, **filter_kwargs(sp_sig.periodogram, kwargs))
		low = np.where(f <= kwargs['cutoff'])[0]
		p = 'semilogy' if log else 'plot'
		figsize = (
			kwargs['fig_x'] * len(px),
			kwargs['fig_y'],
		)
		fig, axes = create_figure(
			nrows=2,
			ncols=len(px),
			figsize=figsize,
			sharex='row',
			sharey='all',
			tight_layout=True,
		)
		for i, x in enumerate(px):
			kws = {
				'lw': kwargs['lw'],
				'color': kwargs['c'],
				'label': f"i = {i}",
			}
			getattr(axes[0, i], p)(f, x, **kws)
			getattr(axes[1, i], p)(f[low], x[low], **kws)
		axes[-1, -1].set_ylim(bottom=kwargs['ylim_bottom'])
		for ax in axes[1, :]:
			ax.set_xlabel('frequency [Hz]')
		for ax in axes[:, 0]:
			ax.set_ylabel('PSD [V**2/Hz]')
		for ax in axes.flat:
			ax.legend(loc='upper right')
		plt.show()
		return f, px

	def _compute_hyperflow(self):
		if not isinstance(self.size, Iterable):
			self.size = (self.size,) * 2
		xl, yl = self.size
		xl = int(np.round(xl / self.sres))
		yl = int(np.round(yl / self.sres))

		xi0 = np.linspace(- xl / 2 + 0.5, xl / 2 - 0.5, xl) * self.sres
		yi0 = np.linspace(- yl / 2 + 0.5, yl / 2 - 0.5, yl) * self.sres
		xi0, yi0 = np.meshgrid(xi0, yi0)

		stim = np.zeros((len(self.opticflow), xl * yl * 2))
		for t in range(len(self.opticflow)):
			xi = xi0 - self.center[t, 0]
			yi = yi0 - self.center[t, 1]
			mask = xi ** 2 + yi ** 2 <= self.radius ** 2
			raw = np.zeros((xl, yl, 2, 6))

			# translation
			raw[..., 0, 0] = mask
			raw[..., 1, 0] = 0
			raw[..., 0, 1] = 0
			raw[..., 1, 1] = mask
			# expansion
			raw[..., 0, 2] = xi * mask
			raw[..., 1, 2] = yi * mask
			# rotation
			raw[..., 0, 3] = -yi * mask
			raw[..., 1, 3] = xi * mask
			# shear 1
			raw[..., 0, 4] = xi * mask
			raw[..., 1, 4] = -yi * mask
			# shear 2
			raw[..., 0, 5] = yi * mask
			raw[..., 1, 5] = xi * mask

			# reconstruct stimuli
			stim[t] = np.reshape(
				a=raw,
				newshape=(-1, raw.shape[-1]),
				order='C',
			) @ self.opticflow[t]

		return stim


class VelField(Obj):
	def __init__(self, x, **kwargs):
		super(VelField, self).__init__(**kwargs)
		self._setup(x)
		self.compute_svd()

	def _setup(self, x: np.ndarray):
		if x.ndim == 4:
			x = np.expand_dims(x, 0)
		assert x.ndim == 5 and x.shape[-1] == 2
		self.num, self.nt, self.nx, self.ny, _ = x.shape
		self.x = x
		self.rho = None
		self.theta = None
		self.maxlag = None
		self.u = None
		self.s = None
		self.v = None
		return

	def get_kers(self, idx: int = 0):
		tker = self.u[..., idx]
		sker = self.v[:, idx, :]
		sker = sker.reshape(
			(self.num, self.nx, self.ny, 2))
		for i in range(self.num):
			maxlag = np.argmax(np.abs(tker[i]))
			if tker[i, maxlag] < 0:
				tker[i] *= -1
				sker[i] *= -1
		return tker, sker

	def compute_svd(
			self,
			x: np.ndarray = None,
			normalize: bool = True, ):
		x = x if x is not None else self.x
		ns = self.nx * self.ny * 2
		u = np.zeros((self.num, self.nt, self.nt))
		s = np.zeros((self.num, self.nt))
		v = np.zeros((self.num, ns, ns))
		for i, a in enumerate(x):
			u[i], s[i], v[i] = sp_lin.svd(
				a.reshape(self.nt, ns))

		max_lags = np.zeros(self.num)
		rho = np.zeros((self.num, self.nx, self.ny))
		theta = np.zeros((self.num, self.nx, self.ny))
		for i in range(self.num):
			tker = u[i, :, 0]
			sker = v[i, 0].reshape(self.nx, self.ny, 2)
			max_lag = np.argmax(np.abs(tker))
			if tker[max_lag] < 0:
				tker *= -1
				sker *= -1
			max_lags[i] = max_lag
			rho[i], theta[i] = vel2polar(sker)

		if normalize:
			s /= s.sum(1, keepdims=True)
			s *= 100

		output = {
			'rho': rho,
			'theta': theta,
			'maxlag': max_lags,
			'u': u,
			's': s,
			'v': v,
		}
		self.setattrs(**output)
		return

	def show(
			self,
			q: float = 0.8,
			idx: int = 0,
			display: bool = True,
			**kwargs, ):
		defaults = {
			'fig_x': 9,
			'fig_y': 1.45,
			'tight_layout': False,
			'title_fontsize': 15,
			'title_y': 1.1,
		}
		for k, v in defaults.items():
			if k not in kwargs:
				kwargs[k] = v
		figsize = (
			kwargs['fig_x'],
			kwargs['fig_y'] * self.num,
		)
		fig, axes = create_figure(
			nrows=self.num,
			ncols=5,
			figsize=figsize,
			sharex='col',
			width_ratios=[2] + [1] * 4,
			tight_layout=kwargs['tight_layout'],
			reshape=True,
		)
		tker, sker = self.get_kers(idx)
		kws1 = {
			'cmap': 'hsv',
			'vmax': 2 * np.pi,
			'vmin': 0,
		}
		for i in range(self.num):
			vminmax = np.max(np.abs(sker[i]))
			kws2 = {
				'cmap': 'bwr',
				'vmax': vminmax,
				'vmin': -vminmax,
			}
			axes[i, 0].plot(tker[i])
			axes[i, 0].axvline(
				self.maxlag[i],
				color='tomato', ls='--', lw=1.2,
				label=f"max lag: {self.maxlag[i] - self.nt}")
			axes[i, 0].legend(fontsize=9)
			axes[i, 1].imshow(self.rho[i], vmin=0)
			x2p = self.rho[i] < np.quantile(self.rho[i].ravel(), q)
			x2p = mwh(x2p, self.theta[i])
			axes[i, 2].imshow(x2p, **kws1)
			axes[i, 3].imshow(sker[i, ..., 0], **kws2)
			axes[i, 4].imshow(sker[i, ..., 1], **kws2)
		titles = [r'$\tau$', r'$\rho$', r'$\theta$', r'$v_x$', r'$v_y$']
		for j, lbl in enumerate(titles):
			axes[0, j].set_title(
				label=lbl,
				y=kwargs['title_y'],
				fontsize=kwargs['title_fontsize'],
			)
		xticks = range(self.nt)
		xticklabels = [
			f"{abs(t - self.nt) * self.tres}"
			if t % 3 == 2 else ''
			for t in xticks
		]
		axes[-1, 0].set(xticks=xticks, xticklabels=xticklabels)
		axes[-1, 0].tick_params(axis='x', rotation=-90, labelsize=8)
		axes[-1, 0].set_xlabel('Time [ms]', fontsize=13)
		add_grid(axes[:, 0])
		remove_ticks(axes[:, 1:], False)
		if display:
			plt.show()
		else:
			plt.close()
		return fig, axes

	def show_full(
			self,
			display: bool = True,
			**kwargs, ):
		defaults = {
			'figsize': (1.25 * self.nt, 6),
			'title_fontsize': 9,
			'title_y': 1.0,
		}
		for k, v in defaults.items():
			if k not in kwargs:
				kwargs[k] = v
		figs = []
		for i, a in enumerate(self.x):
			fig, axes = create_figure(
				nrows=5,
				ncols=self.nt,
				figsize=kwargs['figsize'],
				sharex='all',
				sharey='all',
			)
			rho, theta = vel2polar(a)
			vminmax = np.max(np.abs(a))
			kws1 = {
				'cmap': 'bwr',
				'vmin': -vminmax,
				'vmax': vminmax,
			}
			kws2 = {
				'cmap': 'hsv',
				'vmin': 0,
				'vmax': 2 * np.pi,
			}
			kws3 = {
				'cmap': 'rocket',
				'vmin': np.min(rho),
				'vmax': np.max(rho),
			}
			for t in range(self.nt):
				axes[0, t].imshow(a[t][..., 0], **kws1)
				axes[1, t].imshow(a[t][..., 1], **kws1)
				axes[2, t].imshow(theta[t], **kws2)
				x2p = mwh(rho[t] < 0.3 * np.max(rho), theta[t])
				axes[3, t].imshow(x2p, **kws2)
				axes[4, t].imshow(rho[t], **kws3)
				# title
				time = (t - self.nt) * self.tres
				axes[0, t].set_title(
					label=f't = {t}\n{time}ms',
					fontsize=kwargs['title_fontsize'],
					y=kwargs['title_y'],
				)
			remove_ticks(axes, False)
			figs.append(fig)
			if display:
				plt.show()
			else:
				plt.close()
		return figs


def _expand(arr, reps, axis):
	return np.repeat(np.expand_dims(
		arr, axis=axis),
		repeats=reps,
		axis=axis,
	)


def _check_obj(pos, vel):
	pos = _check_input(pos, -1)
	vel = _check_input(vel, -1)
	assert len(pos) == 3
	assert pos.shape == vel.shape
	assert np.all(np.logical_and(
		pos[-1] > 0,
		pos[-1] <= 1,
	))
	return pos, vel


def _check_input(x, axis):
	if not isinstance(x, np.ndarray):
		x = np.array(x)
	if not x.ndim == 2:
		x = np.expand_dims(x, axis)
	return x
