from .core import *

class Collider():
    def __init__(self, surface: SlideSurface, restitution: float = 0.5, min_penetration: float = 4e-10) -> None:
        self.surface = surface
        self.restitution = restitution
        self.min_penetration = min_penetration

    def collide(self, points: list[Point]) -> None:
        for p in points:
            closest, t = self.surface.closest_point(p.pos)
            offset = p.pos - closest
            normal = self.surface.normal_at(t)
            penetration = np.dot(offset, normal)

            if hasattr(self.surface, "bezier"):
                d = self.surface.tangential_dist(t, p.pos)
                if d > 0: continue

            if penetration >= -self.min_penetration: continue
            else: p.is_grounded = True

            p.pos -= (penetration + self.min_penetration) * normal
            vn = np.dot(p.v, normal)
            if vn < 0:
                p.v -= (1 + self.restitution) * vn * normal

class SoftBody():
    def __init__(self, points: list[Point], damp: float = 0.0, coyote_time: float = 0.1) -> None:
        self.points = points
        self.damp = damp
        self.impulse = np.zeros(2)
        self._now = 0.0
        self._last_ground_time = -1.0
        self.coyote_time = coyote_time
    
    def _apply_forces(self, dt: float) -> None:
        g = np.array([0.0, -9.81])
        for p in self.points:
            if p.w == 0:
                continue
            p.v += dt * g + self.impulse
        self.impulse = np.zeros(2)

    def _integrate(self, dt: float) -> None:
        for p in self.points:
            p.pos += dt * p.v
            p.v *= (1.0 - self.damp)

    def step(self, dt: float) -> None:
        self._apply_forces(dt)
        self._integrate(dt)
    
    def add_impulse(self, impulse: ndarray) -> None:
        self.impulse = impulse
    
    def can_jump(self) -> bool:
        any_ground = any(p.is_grounded for p in self.points)
        if any_ground:
            self._last_ground_time = self._now
            return True
        return (self._now - self._last_ground_time) <= self.coyote_time


class ShapedSoftBody(SoftBody):
    def __init__(self, points: list[Point], colliders: list[Collider], damp: float = 0, rot_damp: float = 0.05, stiffness: float = 0.7) -> None:
        super().__init__(points, damp)
        self.s = stiffness
        self.rot_damp = rot_damp
        self.r = np.array([p.pos.copy() for p in self.points])

        w = np.array([p.m for p in self.points])
        self.mass_sum = np.sum(w)
        self.r_centroid = np.sum(self.r * w[:, None], axis=0) / self.mass_sum

        self.omega = 0.0
        rest_offsets = self.r - self.r_centroid
        self.I_body = np.sum(w * np.sum(rest_offsets**2, axis=1))
        self.I_inv_body = 1.0 / self.I_body

        self.colliders = colliders
    
    def step(self, dt: float) -> None:
        self._init_step(dt)
        self._apply_forces(dt)
        for p in self.points:
                p.pos += p.v * dt

        for collider in self.colliders:
            collider.collide(self.points)

        x_pred = np.array([p.pos for p in self.points])
        w = np.array([p.m for p in self.points])
        c = np.sum(x_pred * w[:, None], axis=0) / self.mass_sum

        R_spin = self._calculate_spin_rot(dt, x_pred, c, w)
        R_shape = self._calculate_shape_rot(x_pred, c, w)
        R_total = R_spin @ R_shape

        self._update_positions(dt, x_pred, c, R_total)

        self._damp_v()

    def _update_positions(self, dt: float, x_pred: ndarray, c: ndarray, R_total: ndarray) -> None:
        inv_dt = 1.0 / dt
        for i, p in enumerate(self.points):
            goal = c + R_total @ (self.r[i] - self.r_centroid)
            correction = self.s * (goal - x_pred[i])
            p.pos += correction
            p.v += correction * inv_dt

    def _calculate_shape_rot(self, x_pred: ndarray, c: ndarray, w: ndarray) -> ndarray:
        A = np.zeros((2, 2))
        for i in range(len(self.points)):
            A += w[i] * np.outer(x_pred[i] - c, self.r[i] - self.r_centroid)
        U, _, Vt = np.linalg.svd(A)
        return U @ Vt
    
    def _calculate_spin_rot(self, dt: float, x_pred: ndarray, c: ndarray, w: ndarray) -> ndarray:
        L = 0.0
        for i, p in enumerate(self.points):
            v_i = (x_pred[i] - p.pos) * (1.0 / dt)
            r = x_pred[i] - c
            L += w[i] * (r[0] * v_i[1] - r[1] * v_i[0])

        self.omega = L * self.I_inv_body
        self.omega *= (1.0 - self.rot_damp)

        theta = self.omega * dt
        ct, st = np.cos(theta), np.sin(theta)
        return np.array([[ct, -st], [st, ct]])

    def _damp_v(self) -> None:
        for p in self.points:
            p.v *= (1.0 - self.damp)
    
    def _init_step(self, dt: float) -> None:
        self._now += dt
        for p in self.points:
            p.is_grounded = False

    def __str__(self):
        c = np.sum([p.pos * p.m for p in self.points], axis=0) / self.mass_sum
        v_mean = np.mean([np.linalg.norm(p.v) for p in self.points])
        return (f"Centroid: ({c[0]:.3f}, {c[1]:.3f}), Angular vel: {self.omega:.3f}, Mean speed: {v_mean:.3f}")


