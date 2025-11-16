import numpy as np
from numpy import ndarray

class Point():
    def __init__(self, pos: ndarray, m: float = 1.0) -> None:
        self.pos = pos
        self.pos_prev = pos.copy()
        self.v = np.zeros_like(pos)
        self.a = np.zeros_like(pos)
        self.m = m
        self.w = 1.0 / self.m

class Constraint():
    def __init__(self, points: list[Point], alpha: float) -> None:
        self.points = points
        self.alpha = alpha
        self.lambda_prev = 0.0

    def constraint(self) -> float: ...
    def grad_vec(self, p: Point) -> ndarray: ...

    def solve(self, dt: float) -> None:
        alpha_tilde = self.alpha / (dt*dt)
        s = sum(p.w * np.dot(self.grad_vec(p), self.grad_vec(p)) for p in self.points) + alpha_tilde
        if s < 1e-12:
            return
        C = self.constraint()
        delta_lambda = -(C + alpha_tilde*self.lambda_prev)/s
        for p in self.points:
            p.pos += delta_lambda * p.w * self.grad_vec(p)
        self.lambda_prev += delta_lambda

class DistConstraint(Constraint):
    def __init__(self, points: list[Point], k: float) -> None:
        super().__init__(points, k)
        self.d0 = np.linalg.norm(points[1].pos - points[0].pos)

    def constraint(self) -> float:
        return np.linalg.norm(self.points[1].pos - self.points[0].pos) - self.d0

    def grad_vec(self, p: Point) -> ndarray:
        p0, p1 = self.points
        d = np.linalg.norm(p1.pos - p0.pos)
        if d < 1e-12:
            return np.zeros_like(p.pos)
        return (p.pos - (p1.pos if p is p0 else p0.pos))/d

class AreaConstraint(Constraint):
    def __init__(self, points: list[Point], k: float):
        super().__init__(points, k)
        self.A0 = self.area()
        self.scale = max(np.linalg.norm(points[i].pos - points[(i+1)%len(points)].pos) for i in range(len(points)))

    def area(self) -> float:
        n = len(self.points)
        return 0.5 * sum(
            self.points[i].pos[0]*self.points[(i+1)%n].pos[1] - 
            self.points[i].pos[1]*self.points[(i+1)%n].pos[0] for i in range(n)
        )

    def constraint(self) -> float:
        return (self.area() - self.A0) / (self.scale ** 2)

    def grad_vec(self, p: Point) -> ndarray:
        n = len(self.points)
        i = self.points.index(p)
        p_prev = self.points[(i-1)%n]
        p_next = self.points[(i+1)%n]
        return 0.5 * np.array([p_prev.pos[1] - p_next.pos[1], p_next.pos[0] - p_prev.pos[0]])

class XPBD():
    def __init__(self, points: list[Point], constraints: list[Constraint], n_sub: int, damp: float = 0.0) -> None:
        self.points = points
        self.C = constraints
        self.n_sub = n_sub
        self.damp = damp
        self.centroid = sum(p.pos for p in points) / len(points)

    def step(self, dt: float) -> None:
        d_sub = dt / self.n_sub
        for _ in range(self.n_sub):
            self.calculate_acceleration()
            for p in self.points:
                p.v += d_sub * p.a
                p.pos_prev = p.pos.copy()
                p.pos += d_sub * p.v
            for c in self.C:
                c.solve(d_sub)
            for p in self.points:
                p.v = (p.pos - p.pos_prev) / d_sub
                p.v *= 1 - self.damp
        self.centroid = sum(p.pos for p in self.points) / len(self.points)

    def calculate_acceleration(self) -> None:
        for p in self.points:
            p.a = np.array([0.0,-9.81])
    
    def __str__(self) -> str:
        return f"XPBD Body centered at ({self.centroid[0]:.3f}, {self.centroid[1]:.3f})"

class SoftBody(XPBD):
    def __init__(self, points: list[Point], n_sub: int, edge_alpha: float, area_alpha: float, diag_alpha: float = 0.0, damp: float = 0.0) -> None:
        constraints = []
        n = len(points)

        for i in range(n):
            constraints.append(DistConstraint([points[i], points[(i+1)%n]], edge_alpha))

        constraints.append(AreaConstraint(points, area_alpha))

        if n > 3 and diag_alpha > 0.0:
            half_n = n // 2
            for i in range(n):
                j = (i + half_n) % n
                constraints.append(DistConstraint([points[i], points[j]], diag_alpha))

        super().__init__(points, constraints, n_sub, damp)


class LineCollider():
    def __init__(self, point: ndarray, normal: ndarray, restitution: float = 1.0) -> None:
        n = np.linalg.norm(normal)
        self.normal = normal / n if n > 1e-12 else np.array([0.0, 1.0])
        self.point = point
        self.restitution = restitution
    
    def collide(self, points: list[Point]) -> None:
        for p in points:
            d = np.dot(p.pos - self.point, self.normal)
            if d < 0.0:
                p.pos -= d * self.normal
                v_n = np.dot(p.v, self.normal)
                p.v -= (1.0 + self.restitution) * v_n * self.normal