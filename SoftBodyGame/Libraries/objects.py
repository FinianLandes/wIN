from .core import *

class LineSlideSurface(SlideSurface):
    def __init__(self, points: ndarray, invis: bool = False) -> None:
        self.invis = invis
        self.points = points if len(points) < 3 else points[0:2]
        tangent = (self.points[1] - self.points[0]) / float(np.linalg.norm(self.points[1] - self.points[0]))
        self.norm = np.array([-tangent[1], tangent[0]])
        self.prev_normal: ndarray | None = None 
        self.blend = 0.08

    def closest_point(self, point: ndarray) -> list[ndarray, float]:
        A, B = self.points
        AB = B - A
        AP = point - A
        t = np.dot(AP, AB) / np.dot(AB, AB)
        t = max(0.0, min(1.0, t))
        return A + t * AB, t
    
    def normal_at(self, t: float) -> ndarray:
        base_normal = -self.norm if self.norm[1] < 0 else self.norm
        if self.prev_normal is not None and t < self.blend:
            t_ratio = t / self.blend
            blended = (1 - t_ratio) * self.prev_normal + t_ratio * base_normal
            norm = np.linalg.norm(blended)
            if norm > 1e-8:
                return blended / norm
            return base_normal
        return base_normal
    
    def tangential_dist(self, t: float, point: ndarray) -> float:
        A = self.points[0]
        tangent = self.points[1] - A
        vec = point - A
        cross = tangent[0] * vec[1] - tangent[1] * vec[0]
        length = np.linalg.norm(tangent)
        if length < 1e-8:
            return 0.0
        return cross / length

class BezierSlideSurface(SlideSurface):
    def __init__(self, points: ndarray,cw: bool = True, invis: bool = False) -> None:
        self.points = points
        self.invis = invis
        self.n = len(self.points) - 1
        self.cw = cw
        self.prev_normal: ndarray | None = None 
        self.blend = 0.08

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
        base_normal = np.array([-tangent[1], tangent[0]])
        if self.cw:
            base_normal = -base_normal

        if self.prev_normal is not None and t < self.blend:
            t_ratio = t / self.blend
            blended = (1 - t_ratio) * self.prev_normal + t_ratio * base_normal
            norm = np.linalg.norm(blended)
            if norm > 1e-8:
                return blended / norm

        return -base_normal if self.cw else base_normal

    def tangential_dist(self, t: float, point: ndarray) -> float:
        t = np.clip(t, 0.0, 1.0)
        B = self.bezier(t)
        tan = self.bezier_deriv(t)
        mag = np.linalg.norm(tan)
        if mag < 1e-8:
            if t < 0.5:
                tan = self.points[1] - self.points[0]
            else:
                tan = self.points[-1] - self.points[-2]
            mag = np.linalg.norm(tan)
            if mag < 1e-8:
                return 0.0
        tan /= mag
        d = np.dot(point - B, tan)
        return d

class TrackGenerator():
    def __init__(self, max_slope: float = 45, min_slope: float = 20, max_gap: float = 3.0, pre_render_range: int = 150, max_seg_len: float = 13, min_seg_len: float = 7, max_len_factor: float = 2.0, use_beziers: bool = False) -> None:
        self.segments: list[SlideSurface] = [LineSlideSurface(np.array([[-20, 10], [10, 0]]))]
        self.max_slope = max_slope #Deg
        self.min_slope = min_slope
        self.pre_render_range = pre_render_range
        self.max_seg_len = max_seg_len
        self.min_seg_len = min_seg_len
        self.max_gap = max_gap
        self.no_gap = True
        self.end_pos = self.segments[0].points[1]
        self.wiggle = 0.8
        self.handle_min = 0.22
        self.handle_max = 0.32
        self.max_len_factor = max_len_factor
        self.use_beziers = use_beziers
        
    
    def update(self, player_pos: ndarray) -> bool:
        mod = False
        if self.end_pos[0] - player_pos[0] < self.pre_render_range:
            self._generate_segment()
            mod = True
                
        while player_pos[0] - self.segments[0].points[-1][0] > self.pre_render_range:
            self.segments.pop(0)
            mod = True
        
        return mod

    def _generate_segment(self) -> None:
        if self.no_gap:
            self.no_gap = False
            self._add_gap()
        else:
            self.no_gap = True

            choice = random.randint(0, 2) if self.use_beziers else 0
            if choice == 0:
                self._add_line()
            else:
                self._add_bezier()
    
    def _calculate_end_point(self) -> ndarray:
        base_len = (random.random() * (self.max_seg_len - self.min_seg_len)) + self.min_seg_len
        length_factor = random.uniform(1.0, self.max_len_factor)
        length = base_len * length_factor

        slope = (random.random() * (self.max_slope - self.min_slope)) + self.min_slope
        dp = np.array([
            math.cos(math.radians(slope)) * length,
            -math.sin(math.radians(slope)) * length
        ])
        return self.end_pos + dp

    def _add_line(self) -> None:
        p1, p2 = self.end_pos, self._calculate_end_point()
        seg = LineSlideSurface(np.array([p1, p2]))
        if self.segments:
            prev_seg = self.segments[-1]
            seg.prev_normal = prev_seg.normal_at(1.0)
        self.segments.append(seg)
        self.end_pos = p2
    
    def _add_bezier(self) -> None:
        p1 = self.end_pos
        p2 = self._calculate_end_point()
        L = p2[0] - p1[0]

        if self.segments:
            prev_normal = self.segments[-1].normal_at(1.0)
            prev_tangent = np.array([prev_normal[1], -prev_normal[0]])
            tangent_len = np.linalg.norm(prev_tangent)
            if tangent_len > 0:
                prev_tangent /= tangent_len
        else:
            prev_tangent = np.array([1.0, 0.0])

        handle1_fraction = random.uniform(self.handle_min, self.handle_max)
        handle1_len = L * handle1_fraction

        c1 = p1 + prev_tangent * handle1_len

        perp = np.array([-prev_tangent[1], prev_tangent[0]])
        wiggle = random.uniform(-self.wiggle, self.wiggle) * math.tan(math.radians(self.max_slope)) * L * 0.12
        c1 += perp * wiggle

        handle2_fraction = random.uniform(self.handle_min, self.handle_max)
        handle2_len = L * handle2_fraction

        max_pull = math.tan(math.radians(self.max_slope)) * L * 0.25
        pull_up = random.uniform(0.0, max_pull)
        kick_down = random.uniform(0.0, max_pull * 0.4)

        c2_y = p2[1] + pull_up + kick_down
        c2 = np.array([p2[0] - handle2_len, c2_y])

        seg = BezierSlideSurface(np.array([p1, c1, c2, p2]))

        if self.segments:
            seg.prev_normal = self.segments[-1].normal_at(1.0)

        self.segments.append(seg)
        self.end_pos = p2.copy()

    def _add_gap(self) -> None:
        self.segments.append(LineSlideSurface(np.array([[self.end_pos[0], self.end_pos[1]], [self.end_pos[0] + self.max_gap, self.end_pos[1]]]), True))
        self.end_pos[0] += self.max_gap

class BgGenerator():
    def __init__(self, chunk_size: float = 20.0, stars_per_chunk: int = 15, pre_render_range: int = 6) -> None:
        self.chunk_size = chunk_size
        self.stars_per_chunk = stars_per_chunk
        self.pre_render_range = pre_render_range
        self.chunks: dict[tuple[int, int], list[Star]] = {}

    def update(self, player_pos: ndarray) -> bool:
        mod = False
        cx = int(np.floor(player_pos[0] / self.chunk_size))
        cy = int(np.floor(player_pos[1] / self.chunk_size))

        needed = set(
            (cx + dx, cy + dy)
            for dx in range(-self.pre_render_range, self.pre_render_range + 1)
            for dy in range(-self.pre_render_range, self.pre_render_range + 1)
        )

        for key in list(self.chunks.keys()):
            if key not in needed:
                del self.chunks[key]
                mod = True

        for key in needed:
            if key not in self.chunks:
                self.chunks[key] = self._generate_chunk(*key)
                mod = True
        return mod

    def _generate_chunk(self, cx: int, cy: int) -> list[Star]:
        rng = np.random.default_rng(hash((cx, cy)) & 0xFFFFFFFF)

        stars = []
        base_x = cx * self.chunk_size
        base_y = cy * self.chunk_size

        for _ in range(self.stars_per_chunk):
            pos = np.array([
                base_x + rng.random() * self.chunk_size,
                base_y + rng.random() * self.chunk_size
            ])
            stars.append(
                Star(
                    pos=pos,
                    size=rng.uniform(1.0, 3.0),
                    luminance=rng.uniform(0.4, 1.0),
                    parallax=rng.uniform(0.0, 1.0),
                    color=(0, 0, rng.uniform(100.0, 255.0))
                )
            )

        return stars
    
    def get_stars(self) -> list[Star]:
        stars = []
        for chunk in self.chunks.values():
            stars.extend(chunk)
        return stars
