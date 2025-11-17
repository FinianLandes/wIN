from .core import *

class LineSlideSurface(SlideSurface):
    def __init__(self, points: ndarray) -> None:
        self.points = points if len(points) < 3 else points[0:2]
        tangent = (self.points[1] - self.points[0]) / float(np.linalg.norm(self.points[1] - self.points[0]))
        self.norm = np.array([-tangent[1], tangent[0]])

    def closest_point(self, point: ndarray) -> list[ndarray, float]:
        A, B = self.points
        AB = B - A
        AP = point - A
        t = np.dot(AP, AB) / np.dot(AB, AB)
        t = max(0.0, min(1.0, t))
        return A + t * AB, t
    
    def normal_at(self, point: ndarray, t: float) -> ndarray:
        A, B = self.points
        tangent = (B - A) / np.linalg.norm(B - A)
        normal = np.array([-tangent[1], tangent[0]])
        if np.dot(point - A, normal) > 0:
            normal = -normal
        return normal

class BezierSlideSurface(SlideSurface):
    def __init__(self, points: ndarray) -> None:
        self.points = points 
        self.n = len(self.points)

    def B(self, n: int, i: int, u: float) -> float:
        return math.comb(n, i) * (1 - u)**(n - i) * u**i

    def bezier(self, t: float) -> ndarray:
        n = self.n - 1
        s = np.zeros(2)
        for i in range(self.n):
            s += self.B(n, i, t) * self.points[i]
        return s

    def bezier_tangent(self, t: float) -> ndarray:
        n = self.n - 2
        s = np.zeros(2)
        for i in range(self.n - 1):
            s += self.B(n, i, t) * (self.points[i+1] - self.points[i]) * (self.n - 1)
        return s

    def closest_t_recursive(self, t0: float, t1: float, p: ndarray, depth: int = 5) -> float:
        if depth == 0:
            ts = np.linspace(t0, t1, 5)
            return min(ts, key=lambda t: np.linalg.norm(p - self.bezier(t)))
        tm = 0.5 * (t0 + t1)
        left = self.closest_t_recursive(t0, tm, p, depth - 1)
        right = self.closest_t_recursive(tm, t1, p, depth - 1)
        return left if np.linalg.norm(p - self.bezier(left)) < np.linalg.norm(p - self.bezier(right)) else right

    def closest_point(self, point: ndarray) -> list[ndarray, float]:
        t = self.closest_t_recursive(0.0, 1.0, point)
        return self.bezier(t), t

    def normal_at(self, point: ndarray, t: float) -> ndarray:
        tangent = self.bezier_tangent(t)
        normal = np.array([-tangent[1], tangent[0]])
        normal /= np.linalg.norm(normal)
        if np.dot(point - self.bezier(t), normal) > 0:
            normal = -normal
        return normal
