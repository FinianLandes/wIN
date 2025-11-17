from .physics import *
from .objects import *

class ObjRender():
    def __init__(self, screen_w: int, screen_h: int, scale: int) -> None:
        self.w = screen_w
        self.h = screen_h
        self.scale = scale
    
    def draw(self, surface: pg.Surface) -> None:
        ...
    
    def to_screen(self, pos: ndarray) -> tuple[int]:
        x = int(pos[0] * self.scale + self.w // 2)
        y = int(self.h - pos[1] * self.scale)
        return (int(x), int(y))

    def in_bounds(self, screen_pos: tuple[int, int]) -> bool:
        x, y = screen_pos
        return 0 <= x < self.w and 0 <= y < self.h

class SoftBodyRender(ObjRender):
    def __init__(self, body: SoftBody, screen_w: int, screen_h: int, scale: float = 100.0, color_rgba: tuple[int] = (255, 200, 100, 80), thickness: int = 3) -> None:
        super().__init__(screen_w, screen_h, scale)
        self.body = body
        self.color = color_rgba
        self.thickness = thickness

    def draw(self, surface: pg.Surface, as_skeleton: bool = False) -> None:
        pts = [self.to_screen(p.pos) for p in self.body.points]
        n = len(pts)

        if as_skeleton:
            for i in range(n):
                p0 = pts[i]
                p1 = pts[(i + 1) % n]
                pg.draw.line(surface, self.color[:3], p0, p1, self.thickness)

            for p in pts:
                pg.draw.circle(surface, self.color[:3], p, 5)
            return

        pg.draw.polygon(surface=surface, color=self.color, points=pts)

        for i in range(n):
            pg.draw.line(surface, self.color[:3], pts[i], pts[(i + 1) % n], self.thickness)

        c = np.mean([p.pos for p in self.body.points], axis=0)
        c_screen = self.to_screen(c)
        pg.draw.circle(surface, (255, 255, 255), c_screen, 4)

class SlideSurfaceRender(ObjRender):
    def __init__(self, slide_surface: SlideSurface, screen_w: int, screen_h: int, scale: float = 100.0, color_rgba: tuple[int] = (100, 200, 100, 80), thickness: int = 3) -> None:
        super().__init__(screen_w, screen_h, scale)
        self.slide_surface = slide_surface
        self.color = color_rgba
        self.thickness = thickness
    
    def draw(self, surface: pg.Surface, as_skeleton: bool = False) -> None:
        if isinstance(self.slide_surface, BezierSlideSurface):
            t_space = np.linspace(0, 1, 50)
            points = [self.to_screen(self.slide_surface.bezier(t)) for t in t_space]
            for i in range(len(points) - 1):
                pg.draw.line(surface, self.color[:3], points[i], points[i + 1], self.thickness)
        else:
            p1, p2 = self.to_screen(self.slide_surface.points[0]), self.to_screen(self.slide_surface.points[1])
            pg.draw.line(surface, self.color[:3], p1, p2, self.thickness)
