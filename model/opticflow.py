from .utils_model import *


_CHOICES = ['obj', 'transl', 'fixate', 'pursuit']


_Obj = collections.namedtuple(
	typename='Object',
	field_names=[
		'pos', 'v',      # coordinate system: real
		'alpha', 'r',  # coordinate system: self
		'size'],
)


class OpticFlow(object):
	def __init__(
			self,
			category: str,
			num: int = 1,
			n_obj: int = 1,
			fov: float = 45,
			res: float = 1.0,
			obj_r: float = 0.3,
			obj_bound: float = 0.97,
			obj_zlim: Tuple[float, float] = (0.324, 1),
			vlim_slf: Tuple[float, float] = (0.01, 1.0),
			vlim_obj: Tuple[float, float] = (0.01, 2.0),
			z_bg: float = 1.0,
			seed: int = 0,
	):
		super(OpticFlow, self).__init__()
		assert category in _CHOICES,\
			f"allowed categories:\n{_CHOICES}"
		assert isinstance(num, int) and num > 0
		assert isinstance(n_obj, int) and n_obj >= 0
		assert isinstance(z_bg, float) and z_bg > 0
		if category in ['obj', 'pursuit']:
			assert n_obj > 0
		self.category = category
		self.num = num
		self.n_obj = n_obj
		self.fov = fov
		self.res = res
		self.z_bg = z_bg
		self.obj_r = obj_r
		self.obj_zlim = obj_zlim
		self.obj_bound = obj_bound
		self.vlim_slf = vlim_slf
		self.vlim_obj = vlim_obj
		self.rng = get_rng(seed)
		self._init_span()
		self._init_polar_coords()
		self.alpha_dot = None
		self.v_slf = None
		self.objects = {}

	def compute_flow(self):
		if self.category == 'obj':
			self.v_slf = np.zeros((self.num, 3))
		else:
			self.v_slf = self.sample_vel(*self.vlim_slf)
		v_tr, x_obj = self._add_objects()
		v_tr -= self._prepare_v(self.v_slf)
		v_rot = self._compute_v_rot(x_obj)
		self.alpha_dot = compute_alpha_dot(
			v=v_tr - v_rot, x=x_obj, axis=3)
		return x_obj, v_tr, v_rot

	def compute_coords(self, fix: np.ndarray = None):
		if fix is None:
			fix = self.sample_fix()
		self._compute_fix(fix)
		self._compute_rot()
		self._compute_xyz()
		return self

	def sample_vel(self, vmin: float, vmax: float):
		v = self.rng.normal(size=(self.num, 3))
		v /= sp_lin.norm(v, axis=-1, keepdims=True)
		v *= self.rng.uniform(
			low=vmin,
			high=vmax,
			size=(self.num, 1),
		)
		return v

	def sample_pos(self, flat_criterion: bool = True):
		bound = self.obj_bound * self.fov
		if flat_criterion:
			bound = np.tan(np.deg2rad(bound))
		pos = np_nans((self.num, 3))
		for i in range(self.num):
			while True:
				u = self.rng.normal(size=3)
				if flat_criterion:
					_, th, ph = cart2polar(
						self.R[i].T @ u).ravel()
					a = np.tan(th) * np.cos(ph)
					b = np.tan(th) * np.sin(ph)
					cond = (
						np.abs(a) < bound and
						np.abs(b) < bound
					)
				else:
					c = sp_dist.cosine(self.fix[i], u)
					c = np.rad2deg(np.arccos(1 - c))
					cond = c < bound
				if cond:
					break
			_, th, ph = cart2polar(u).ravel()
			z = self.rng.uniform(
				low=self.obj_zlim[0],
				high=self.obj_zlim[1],
			)
			pos[i] = polar2cart(np.array(
				[z / np.cos(th), th, ph]
			)).ravel()
		assert not np.isnan(pos).sum()
		return pos

	def sample_fix(self):
		fix = np_nans((self.num, 2))
		bound = 1 / np.tan(np.deg2rad(self.fov))
		kws = dict(low=-bound, high=bound)
		i = 0
		while True:
			x = self.rng.uniform(**kws)
			y = self.rng.uniform(**kws)
			if abs(x) + abs(y) < 1:
				fix[i] = x, y
				i += 1
			if i == self.num:
				break
		assert not np.isnan(fix).sum()
		return fix

	def apply_rot(
			self,
			arr: np.ndarray,
			transpose: bool, ):
		etc = ''.join(list(map(
			chr, range(98, 123)
		)))[:arr.ndim - 2]
		operands = 'aij, '
		operands += ' -> '.join([
			'a' + etc + 'i' if transpose else 'j',
			'a' + etc + 'j' if transpose else 'i',
		])
		return np.einsum(operands, self.R, arr)

	def compute_obj_mask(self, pos: np.ndarray):
		mask = self.x_bg - pos.reshape((self.num, 1, 1, 3))
		mask = sp_lin.norm(mask[..., :2], axis=-1) < self.obj_r
		return mask

	def _add_objects(self):
		# pos:        real coordinate system
		# pos_self:   self coordinate system
		if self.n_obj == 0:
			return 0, self.x
		obj_masks = {}
		for obj_i in range(1, self.n_obj + 1):
			if self.category == 'pursuit' and obj_i == 1:
				pos = self.fix.copy()
				pos[:, 2] = self.rng.uniform(
					low=self.obj_zlim[0],
					high=self.obj_zlim[1],
				)
			else:
				pos = self.sample_pos(flat_criterion=True)
			pos_self = self.apply_rot(pos, transpose=True)
			mask = self.compute_obj_mask(pos)
			kws = dict(
				pos=pos,
				v=self.sample_vel(*self.vlim_obj),
				alpha=np.stack([
					np.arctan2(pos_self[:, 0], pos_self[:, 2]),
					np.arctan2(pos_self[:, 1], pos_self[:, 2]),
				], axis=1),
				r=cart2polar(pos_self),
				size=mask.mean(-1).mean(-1),
			)
			self.objects[obj_i] = _Obj(**kws)
			obj_masks[obj_i] = mask
		x_obj, v_tr = self._compute_obj_v(obj_masks)
		return v_tr, x_obj

	def _compute_v_rot(self, x: np.ndarray):
		if self.category in ['obj', 'transl']:
			return 0
		if self.category == 'fixate':
			v = -self.v_slf
		elif self.category == 'pursuit':
			v = self.objects[1].v - self.v_slf
		v_rot = None  # TODO: fix this
		return v_rot

	def _compute_obj_v(self, obj_masks: dict):
		x_obj = self.x_bg.copy()
		v_tr = np.zeros((self.num, self.dim, self.dim, 3))
		for i in range(self.num):
			obj_z = {
				obj_i: obj.pos[i, 2] for
				obj_i, obj in self.objects.items()
			}
			order, obj_z = zip(*sorted(
				obj_z.items(),
				key=lambda t: t[1],
				reverse=True,
			))
			for obj_i, z in zip(order, obj_z):
				m = obj_masks[obj_i][i]
				x_obj[i, ..., -1][m] = z
				for j in range(3):
					v = self.objects[obj_i].v[i, j]
					v_tr[i, ..., j][m] = v
		x_obj = self.apply_rot(x_obj, transpose=True)
		v_tr = self.apply_rot(v_tr, transpose=True)
		return x_obj, v_tr

	def _prepare_v(self, v: np.ndarray):
		v_transl = self.apply_rot(v, transpose=True)
		v_transl = _expand(v_transl, self.dim, 1)
		v_transl = _expand(v_transl, self.dim, 1)
		return v_transl

	def _compute_fix(self, fix: np.ndarray):
		fix = _check_input(fix, 0)
		assert fix.shape[1] == 2, "fix = (X0, Y0)"
		upper = 1 / np.tan(np.deg2rad(self.fov))
		okay = np.abs(fix).sum(1) < upper
		fix_z = self.z_bg * np.ones(
			(okay.sum(), 1))
		self.fix = np.concatenate([
			fix[okay],
			fix_z,
		], axis=-1)
		return

	def _compute_xyz(self):
		gamma = 'aij, mnj -> amni'
		gamma = np.einsum(gamma, self.R, self.tan)
		shape = (1, self.dim, self.dim, 1)
		# x: measured in self coordinate system:
		self.x = self.z_bg * np.concatenate([
			self.tan[..., 0].reshape(shape),
			self.tan[..., 1].reshape(shape),
			np.ones(shape),
		], axis=-1) / gamma[..., [-1]]
		# x_bg: measured in real coordinate system:
		x_bg = 'aij, amnj -> amni'
		self.x_bg = np.einsum(x_bg, self.R, self.x)
		assert np.all(np.round(self.x_bg[..., -1], 10) == self.z_bg)
		return

	def _compute_rot(self):
		r = cart2polar(self.fix)
		u0 = np.concatenate([
			- np.sin(r[:, [2]]),
			np.cos(r[:, [2]]),
			np.zeros((len(r), 1)),
		], axis=-1)
		self.R = Rotation.from_rotvec(
			r[:, [1]] * np.array(u0),
		).as_matrix()
		return

	def _init_polar_coords(self):
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

	def _init_span(self):
		dim = self.fov / self.res
		assert dim.is_integer()
		self.dim = 2 * int(dim) + 1
		self.span = np.deg2rad(np.linspace(
			-self.fov, self.fov, self.dim))
		return


class Simulation(object):
	def __init__(
			self,
			num: int,
			fov: float,
			res: float,
			seed: int = 0,
			ratio: float = 0.8,
			verbose: bool = False,
	):
		super(Simulation, self).__init__()
		self.num = num
		self.fov = fov
		self.res = res
		self.ratio = ratio
		self.verbose = verbose
		self.rng = get_rng(seed)
		self.fix = None
		self.vel_slf = None
		self.vel_obj = None
		self.pos_obj = None
		self.alpha_dot = None

	def fit_sim(self, parallel: bool = True):
		if parallel:
			with joblib.parallel_backend('multiprocessing'):
				alpha_dot = joblib.Parallel(n_jobs=-1)(
					joblib.delayed(of_fit_single)(
						self.fov,
						self.res,
						self.fix[i],
						self.vel_slf[:, i],
						self.vel_obj[:, i],
						self.pos_obj[:, i],
					) for i in range(self.num)
				)
		else:
			alpha_dot = []
			for i in range(self.num):
				alpha_dot.append(of_fit_single(
					self.fov,
					self.res,
					self.fix[i],
					self.vel_slf[:, i],
					self.vel_obj[:, i],
					self.pos_obj[:, i],
				))
		alpha_dot = np.concatenate(alpha_dot)
		assert not np.isnan(alpha_dot).sum()
		self.alpha_dot = alpha_dot
		return self

	def sample(self, n: int = None):
		n = n if n else self.num
		fix = np_nans((n, 2))
		pos_obj = np_nans((3, n))
		vel_slf = np_nans((3, n))
		vel_obj = np_nans((3, n))
		for i in range(n):
			fix[i] = self._sample_fix()
			pos_obj[:, i] = self._sample_pos(fix[i])
			vel_slf[:, i] = self._sample_vel(0.01, 1)
			vel_obj[:, i] = self._sample_vel(0.01, 2)
		self.fix = fix
		self.vel_slf = vel_slf
		self.vel_obj = vel_obj
		self.pos_obj = pos_obj
		return self

	def _of_fit(self, i: int):
		of = OpticFlowVec(self.fov, self.res).compute_coords(self.fix[i])
		x = of.compute_flow(self.vel_slf[:, i], self.pos_obj[:, i], self.vel_obj[:, i])
		x = x[..., 0, 0]
		return x

	def _sample_fix(self):
		bound = 1 / np.tan(np.deg2rad(self.fov))
		kws = dict(low=-bound, high=bound)
		while True:
			x = self.rng.uniform(**kws)
			y = self.rng.uniform(**kws)
			if abs(x) + abs(y) < 1:
				fix = (x, y)
				break
		return fix

	def _sample_pos(self, fix, z=(0.5, 1)):
		f = np.append(fix, 1)
		while True:
			e = self.rng.normal(size=3)
			d = sp_dist.cosine(f, e)
			d = np.rad2deg(np.arccos(1 - d))
			if d < self.ratio * self.fov:
				break
		_, th, ph = cart2polar(e).ravel()
		z = self.rng.uniform(low=z[0], high=z[1])
		pos = polar2cart(np.array([z/np.cos(th), th, ph]))
		return pos.ravel()

	def _sample_vel(self, vmin, vmax):
		v = self.rng.normal(size=3)
		v /= sp_lin.norm(v)
		v *= self.rng.uniform(
			low=vmin, high=vmax)
		return v


class SimulationMult(object):
	def __init__(
			self,
			fov: float,
			n_fix: int,
			n_slf: int,
			n_obj: int,
			seed: int = 0,
			verbose: bool = False,
	):
		super(SimulationMult, self).__init__()
		self.fov = fov
		self.n_fix = n_fix
		self.n_slf = n_slf
		self.n_obj = n_obj
		self.verbose = verbose
		self.rng = get_rng(seed)
		self.fix = None
		self.vel_slf = None
		self.vel_obj = None
		self.pos_obj = None
		self.acc = None
		self._idxs()

	def uniform(
			self,
			vel_slf: Tuple[float, float] = (0.01, 1),
			vel_obj: Tuple[float, float] = (0.01, 2),
	):
		self.uniform_fix(self.n_fix)
		self.vel_slf = self.uniform_vel(
			self.n_slf, vel_slf[0], vel_slf[1])
		self.vel_obj = self.uniform_vel(
			self.n_obj, vel_obj[0], vel_obj[1])
		self.uniform_pos(self.n_obj)
		self._accept()
		return self

	def uniform_fix(self, n: int):
		fix = np_nans((n, 2))

		bound = 1 / np.tan(np.deg2rad(self.fov))
		kws = dict(low=-bound, high=bound)

		idx = 0
		while True:
			x = self.rng.uniform(**kws)
			y = self.rng.uniform(**kws)
			if abs(x) + abs(y) < 1:
				fix[idx] = x, y
				idx += 1
			if idx == n:
				break
		assert not np.isnan(fix).sum()
		self.fix = fix
		return

	def uniform_sphere(self, n: int):
		points = self.rng.normal(size=(3, n))
		points /= sp_lin.norm(
			a=points,
			axis=0,
			keepdims=True,
		)
		return points

	def uniform_vel(
			self,
			n: int,
			vmin: float,
			vmax: float, ):
		v = self.uniform_sphere(n)
		v *= self.rng.uniform(
			low=vmin,
			high=vmax,
			size=(1, n),
		)
		return v

	def uniform_pos(
			self,
			n: int,
			x: Tuple[float, float] = (-1, 1),
			y: Tuple[float, float] = (-1, 1),
			z: Tuple[float, float] = (0.5, 1), ):
		pos = [
			self.rng.uniform(
				low=e[0],
				high=e[1],
				size=n,
			) for e in [x, y, z]
		]
		self.pos_obj = np.stack(pos, axis=0)
		return

	def _accept(self, fov_ratio: float = 0.8, z: float = 1):
		fix = np.concatenate([
			self.fix,
			z * np.ones((len(self.fix), 1)),
		], axis=1)
		d = sp_dist.cdist(
			XA=fix,
			XB=self.pos_obj.T,
			metric='cosine',
		)
		theta = np.rad2deg(np.arccos(1 - d))
		acc = theta < fov_ratio * self.fov
		acc = _expand(acc, self.vel_slf.shape[1], 1)
		self.acc = acc.ravel()
		if self.verbose:
			msg = f"{100 * self.acc.sum() / len(self.acc):0.1f} % of total"
			msg += f" simulations accepted (using fov_ratio = {fov_ratio})"
			print(msg)
		return

	def _idxs(self):
		self.idxs = {
			i: (a, b, c) for i, (a, b, c) in
			enumerate(itertools.product(
				range(self.n_fix),
				range(self.n_slf),
				range(self.n_obj),
			))
		}
		return


class OpticFlowVec(object):
	def __init__(
			self,
			fov: float = 45,
			res: float = 0.1,
			obj_r: float = 0.2,
			z_bg: float = 1,
	):
		super(OpticFlowVec, self).__init__()
		assert z_bg > 0
		self.fov = fov
		self.res = res
		self.z_bg = z_bg
		self.obj_r = obj_r
		self._compute_span()
		self._create_ticks()
		self._compute_polar_coords()

	def compute_flow(
			self,
			vel: np.ndarray,
			obj_pos: np.ndarray,
			obj_vel: np.ndarray, ):
		self._set_vals(vel, obj_pos, obj_vel)
		# add object
		v_transl_obj, x_obj = self._add_obj()
		x = _expand(self.x, self.obj_pos.shape[1], -1)
		x[~np.isnan(x_obj)] = x_obj[~np.isnan(x_obj)]
		# apply self movement
		v_rot = self._compute_v_rot(x=x)
		v_transl = self._compute_v_tr()
		v_transl = _expand(v_transl, x.shape[-1], -1)
		# expand/merge together
		kws = dict(reps=self.vel.shape[1], axis=-2)
		x = _expand(x, **kws)
		nans = _expand(np.isnan(x_obj), **kws)
		v_transl_obj = _expand(v_transl_obj, **kws)
		v_transl[~nans] += v_transl_obj[~nans]
		# compute retinal velocity
		alpha_dot = self._compute_alpha_dot(
			v=v_transl - v_rot, x=x, axis=3)
		return alpha_dot

	def compute_coords(self, fix: np.ndarray = (0, 0)):
		self._compute_fix(fix)
		self._compute_rot()
		self._compute_xyz()
		return self

	def _set_vals(self, vel, obj_pos, obj_vel):
		self.obj_pos, self.obj_vel = _check_obj(
			obj_pos, obj_vel)
		self.vel = _check_input(vel, -1)
		return

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
		self.R = Rotation.from_rotvec(
			r[:, [1]] * np.array(u0),
		).as_matrix()
		return

	def _compute_fix(self, fix: np.ndarray):
		fix = _check_input(fix, 0)
		assert fix.shape[1] == 2, "fix = (X0, Y0)"
		upper = 1 / np.tan(np.deg2rad(self.fov))
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

	def _create_ticks(self, tick_spacing: int = None):
		if tick_spacing is None:
			tick_spacing = 15 // self.res  # self.fov // 3
		self.ticks, self.ticklabels = zip(*[
			(i, str(int(np.round(np.rad2deg(x))))) for i, x
			in enumerate(self.span) if i % tick_spacing == 0
		])
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


def of_fit_single(fov, res, fix, vel_self, vel_obj, pos_obj):
	of = OpticFlowVec(fov, res).compute_coords(fix)
	x = of.compute_flow(vel_self, pos_obj, vel_obj)
	return x[..., 0, 0]


def compute_alpha_dot(
		v: np.ndarray,
		x: np.ndarray,
		axis: int = 3, ):
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


def vself2polar(
		a: np.ndarray,
		b: np.ndarray,
		dtype=float, ):
	ta = np.tan(a, dtype=dtype)
	tb = np.tan(b, dtype=dtype)
	theta = np.sqrt(ta**2 + tb**2)
	theta = np.arctan(theta, dtype=dtype)
	phi = np.arctan2(tb, ta, dtype=dtype)
	phi[phi < 0] += 2 * np.pi
	return theta, phi


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
