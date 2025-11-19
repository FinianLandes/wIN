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
    
    def normal_at(self, t: float) -> ndarray:
        return self.norm

class BezierSlideSurface(SlideSurface):
    def __init__(self, points: ndarray, cw: bool = False) -> None:
        self.points = points 
        self.n = len(self.points) - 1
        self.cw = cw

    def _newton(self, init_t: float, point: ndarray, n_steps: int = 5) -> float:
        t = init_t
        for _ in range(6):
            p = self.bezier(t)
            dp = self.bezier_deriv(t)
            r = point - p
            proj = np.dot(r, dp)
            denom = np.dot(dp, dp)
            if denom < 1e-12:
                break
            t += proj / denom
            t = float(np.clip(t, 0.0, 1.0))
        return t
    
    def _bernstein(self, i: int, t: float) -> float:
        return math.comb(self.n, i) * (1 - t)**(self.n - i) * t**i

    def bezier(self, t: float) -> ndarray:
        p = np.zeros(2)
        for i, pt in enumerate(self.points):
            p += self._bernstein(i, t) * pt
        return p

    def bezier_deriv(self, t: float) -> ndarray:
        d = np.zeros(2)
        for i in range(self.n):
            d += self._bernstein(i, t) * (self.points[i+1] - self.points[i])
        return d * self.n

    def closest_point(self, point: ndarray, samples: int = 64) -> tuple[ndarray, float]:
        ts = np.linspace(0.0, 1.0, samples)
        d2 = []
        for t in ts:
            p = self.bezier(t)
            d2.append(np.dot(point - p, point - p))
        i = int(np.argmin(d2))
        t = ts[i]

        if i == 0: return self.points[0], 0.0
        if i == samples - 1: return self.points[-1], 1.0

        t = self._newton(t, point)

        if t <= 0.0: return self.points[0], 0.0
        if t >= 1.0: return self.points[-1], 1.0

        return self.bezier(t), t

    def normal_at(self, t: float) -> ndarray:
        t = np.clip(t, 0.0, 1.0)
        tangent = self.bezier_deriv(t)
        mag = np.linalg.norm(tangent)
        if mag < 1e-8:
            if t < 0.5:
                tangent = self.points[1] - self.points[0]
            else:
                tangent = self.points[-1] - self.points[-2]
            mag = np.linalg.norm(tangent)
        tangent /= max(mag, 1e-8)
        normal = np.array([-tangent[1], tangent[0]])
        return -normal if self.cw else normal

    def tangential_dist(self, t: float, point: ndarray) -> float:
        if t < 1e-4:
            B = self.points[0]
            tan = self.bezier_deriv(0.00001)
        elif t > 1 - 1e-4:
            B = self.points[-1]
            tan = self.bezier_deriv(0.99999)
        else:
            return -1.0
        tan_norm = tan / np.linalg.norm(tan)
        d = point - B
        return np.dot(d, tan_norm)
        
