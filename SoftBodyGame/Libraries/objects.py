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
        if self.norm[1] < 0:
            return -self.norm
        return self.norm

class BezierSlideSurface(SlideSurface):
    def __init__(self, points: ndarray, cw: bool = False) -> None:
        self.points = points 
        self.n = len(self.points) - 1
        self.cw = cw

    def _newton(self, init_t: float, point: ndarray, n_steps: int = 5) -> float:
        t = init_t
        for _ in range(n_steps):
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
            tan = self.bezier_deriv(1e-4)
        elif t > 1 - 1e-4:
            B = self.points[-1]
            tan = self.bezier_deriv(1 - 1e-4)
        else:
            return -1.0
        tan_norm = tan / np.linalg.norm(tan)
        d = point - B
        return np.dot(d, tan_norm)


class TrackGenerator():
    def __init__(self, max_slope: float = 45, min_slope: float = 10, max_gap: float = 1.0, pre_render_range: int = 100, max_cont_seg: int = 3, max_seg_len: float = 10, min_seg_len: float = 5) -> None:
        self.segments: list[SlideSurface] = [LineSlideSurface(np.array([[-20, 10], [10, 5]]))]
        self.max_slope = max_slope #Deg
        self.min_slope = min_slope
        self.pre_render_range = pre_render_range
        self.max_cont_seg = max_cont_seg
        self.max_seg_len = max_seg_len
        self.min_seg_len = min_seg_len
        self.max_gap = max_gap
        self.no_gap_count = 0
        self.end_pos = self.segments[0].points[1]
        
    
    def update(self, player_pos: ndarray) -> None:
        if self.end_pos[0] - player_pos[0] < self.pre_render_range:
            #self._generate_segment()
            ...
                
        while player_pos[0] - self.segments[0].points[-1][0] > self.pre_render_range:
            self.segments.pop(0)

    def _generate_segment(self) -> None:
        if self.no_gap_count > self.max_cont_seg:
            self.no_gap_count = 0
            self._add_gap()
        else:
            if self.no_gap_count == 0:
                choice = random.randint(0, 1)
            else: 
                choice = random.randint(0, 2)
            match choice:
                case 0:
                    self._add_line()
                    self.no_gap_count += 1
                case 1:
                    self._add_bezier()
                    self.no_gap_count += 1
                case 2:
                    self._add_gap()
                    self.no_gap_count = 0
    
    def _calculate_end_point(self) -> ndarray:
        slope = (random.random() * (self.max_slope - self.min_slope)) + self.min_slope
        length = (random.random() * (self.max_seg_len - self.min_seg_len)) + self.min_seg_len
        dp = np.array([math.cos(math.radians(slope)) * length, -math.sin(math.radians(slope)) * length])
        return self.end_pos + dp

    def _add_line(self) -> None:
        p1, p2 = self.end_pos, self._calculate_end_point()
        seg = LineSlideSurface(np.array([p2, p1]))
        self.segments.append(seg)
        self.end_pos = p2
    
    def _add_bezier(self) -> None:
        p1, p2 = self.end_pos, self._calculate_end_point()
        L = p2[0] - p1[0]
        max_height = math.tan(math.radians(self.max_slope)) * L
        random_offset1, random_offset2 = random.uniform(-max_height, max_height), random.uniform(-max_height, max_height)
        slope1, slope2 = random.uniform(-0.3, -0.002), random.uniform(-0.3, -0.002)
        c1 = p1[0] + L * 0.25, p1[1] + slope1 * L * 0.25 + random_offset1
        c2 = p1[0] + L * 0.75, p2[1] + slope2 * L * 0.75 + random_offset2
        seg = BezierSlideSurface(np.array([p1, c1, c2, p2]))
        self.segments.append(seg)
        self.end_pos = p2

    def _add_gap(self) -> None:
        self.end_pos[0] += self.max_gap


