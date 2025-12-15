from .physics import *
from .objects import *

class ObjRender():
    def __init__(self, screen_w: int, screen_h: int, scale: int, static: bool = False, open_gl: bool = False) -> None:
        self.w = screen_w
        self.h = screen_h
        self.scale = scale
        self.cam_x = 0.0
        self.cam_y = 0.0
        self.static = static
        self.open_gl = open_gl
    
    def draw(self, surface: pg.Surface, as_skeleton: bool = False) -> None:
        if self.open_gl:
            self._draw_gl(as_skeleton)
        else:
            self._draw_pg(surface, as_skeleton)

    def _draw_gl(self, as_skeleton: bool = False) -> None:
        ...
    def _draw_pg(self, surface: pg.Surface, as_skeleton: bool = False) -> None:
        ...
    
    def update_camera(self, player_pos: ndarray) -> None:
        if not self.static:
            self.cam_x = player_pos[0]
            self.cam_y = player_pos[1]

    def to_screen(self, pos: ndarray) -> list[int]:
        dx = (pos[0] - self.cam_x) * self.scale
        dy = (pos[1] - self.cam_y) * self.scale

        x = dx + self.w * 0.5
        y = self.h * 0.5 - dy
        return [int(x), int(y)]

    def in_bounds(self, screen_pos: tuple[int, int]) -> bool:
        x, y = screen_pos
        return 0 <= x < self.w and 0 <= y < self.h

class SoftBodyRender(ObjRender):
    def __init__(self, body: SoftBody, screen_w: int, screen_h: int, scale: float = 100.0, color_rgba: tuple[int] = (255, 200, 100, 80), thickness: int = 3, glow: int = 3, static: bool = False, open_gl: bool = False) -> None:
        super().__init__(screen_w, screen_h, scale, static, open_gl)
        self.body = body
        self.color = color_rgba
        self.thickness = thickness
        self.glow = glow
    
    def _draw_gl(self, as_skeleton: bool = False) -> None:
        pts = []
        for p in self.body.points:
            p = self.to_screen(p.pos)
            p[1] -= self.thickness
            pts.append(p)
        r, g, b, a = [c / 255.0 for c in self.color]

        def begin_cb(mode) -> None:
            glBegin(mode)

        def end_cb() -> None:
            glEnd()

        def vertex_cb(vertex: list, data = None) -> None:
            glVertex2f(vertex[0], vertex[1])

        def combine_cb(coords: list, vertex_data, weight) -> list:
            return [coords[0], coords[1], 0.0]

        def error_cb(err: GLUError) -> None:
            print("Tessellation error:", gluErrorString(err))

        tess = gluNewTess()
        gluTessCallback(tess, GLU_TESS_BEGIN, begin_cb)
        gluTessCallback(tess, GLU_TESS_END, end_cb)
        gluTessCallback(tess, GLU_TESS_VERTEX, vertex_cb)
        gluTessCallback(tess, GLU_TESS_ERROR, error_cb)
        gluTessCallback(tess, GLU_TESS_COMBINE, combine_cb)

        glColor4f(r, g, b, a * 0.8)
        gluTessBeginPolygon(tess, None)
        gluTessBeginContour(tess)
        for x, y in pts:
            gluTessVertex(tess, [x, y, 0.0], [x, y, 0.0])
        gluTessEndContour(tess)
        gluTessEndPolygon(tess)
        gluDeleteTess(tess)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        for i in range(self.glow):
            alpha = a * (0.15 / (i + 1))
            glColor4f(r, g, b, alpha)
            glLineWidth(self.thickness + i * 2)
            glBegin(GL_LINE_LOOP)
            for x, y in pts:
                glVertex2f(x, y)
            glEnd()
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glColor4f(r, g, b, a)
        glLineWidth(self.thickness)
        glBegin(GL_LINE_LOOP)
        for x, y in pts:
            glVertex2f(x, y)
        glEnd()

        if as_skeleton:
            glPointSize(6)
            glBegin(GL_POINTS)
            for x, y in pts:
                glVertex2f(x, y)
            glEnd()
    
    def _draw_pg(self, surface: pg.Surface, as_skeleton: bool = False) -> None:
        pts = []
        for p in self.body.points:
            p = self.to_screen(p.pos)
            p[1] -= self.thickness
            pts.append(p)
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

class SlideSurfaceRender(ObjRender):
    def __init__(self, slide_surface: SlideSurface, screen_w: int, screen_h: int, scale: float = 100.0, color_rgba: tuple[int] = (100, 200, 100, 80), thickness: int = 3, glow: int = 3, static: bool = False, open_gl: bool = True,) -> None:
        super().__init__(screen_w, screen_h, scale, static, open_gl)
        self.slide_surface = slide_surface
        self.color = color_rgba
        self.thickness = thickness
        self.glow = glow
    
    def _draw_gl(self, as_skeleton: bool = False) -> None:
        r, g, b, a = [c / 255.0 for c in self.color]

        if isinstance(self.slide_surface, BezierSlideSurface):
            t_space = np.linspace(0, 1, 50)
            pts = [self.to_screen(self.slide_surface.bezier(t)) for t in t_space]
        else:
            pts = [
                self.to_screen(self.slide_surface.points[0]),
                self.to_screen(self.slide_surface.points[1])
            ]

        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

        for i in range(self.glow):
            glColor4f(r, g, b, a * (0.2 / (i + 1)))
            glLineWidth(self.thickness + i * 2)

            glBegin(GL_LINE_STRIP)
            for x, y in pts:
                glVertex2f(x, y)
            glEnd()
        
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glColor4f(r, g, b, a)
        glLineWidth(self.thickness)

        glBegin(GL_LINE_STRIP)
        for x, y in pts:
            glVertex2f(x, y)
        glEnd()

        if as_skeleton:
            pm = self.slide_surface.bezier(0.5) if hasattr(self.slide_surface, "bezier") else \
                (self.slide_surface.points[0] + self.slide_surface.points[1]) * 0.5
            nrm = self.slide_surface.normal_at(0.5)

            p0 = self.to_screen(pm)
            p1 = self.to_screen(pm + nrm)

            glLineWidth(2)
            glBegin(GL_LINES)
            glVertex2f(*p0)
            glVertex2f(*p1)
            glEnd()
    
    def _draw_pg(self, surface: pg.Surface, as_skeleton: bool = False) -> None:
        if isinstance(self.slide_surface, BezierSlideSurface):
            t_space = np.linspace(0, 1, 50)
            points = [self.to_screen(self.slide_surface.bezier(t)) for t in t_space]
            for i in range(len(points) - 1):
                pg.draw.line(surface, self.color[:3], points[i], points[i + 1], self.thickness)
            if as_skeleton:
                pm = self.slide_surface.bezier(0.5)
                norm = self.slide_surface.normal_at(0.5)
                pe = self.to_screen(pm + norm)
                pm = self.to_screen(pm)
                pg.draw.line(surface, self.color[:3], pm, pe, self.thickness)
        else:
            p1, p2 = self.to_screen(self.slide_surface.points[0]), self.to_screen(self.slide_surface.points[1])
            pg.draw.line(surface, self.color[:3], p1, p2, self.thickness)
            if as_skeleton:
                pm = (self.slide_surface.points[0] + self.slide_surface.points[1]) / 2
                norm = self.slide_surface.normal_at(0.5)
                pe = self.to_screen(pm + norm)
                pm = self.to_screen(pm)
                pg.draw.line(surface, self.color[:3], pm, pe, self.thickness)

class TextRender(ObjRender):
    def __init__(self, text: str, screen_w: int, screen_h: int, scale: int, font: pg.font.Font, color_rgba: tuple[int] = (0, 200, 100, 80), pos: ndarray = np.array([0.0, 0.0]), glow: int = 3,  static: bool = False, open_gl: bool = True) -> None:
        super().__init__(screen_w, screen_h, scale, static, open_gl)
        self.font = font
        self.pos = pos
        self.text = text
        self.color = color_rgba
        self.glow = glow
    
    def draw(self, surface: pg.Surface, as_skeleton: bool = False, text: str | None = None) -> None:
        if self.open_gl:
            self._draw_gl(as_skeleton, text)
        else:
            self._draw_pg(surface, as_skeleton)
    
    def _draw_gl(self, as_skeleton: bool = False, text: str | None = None) -> None:
        if text:
            self.text = text

        rendered = self.font.render(self.text, True, self.color[:3])
        w, h = rendered.get_size()
        tex_data = pg.image.tostring(rendered, "RGBA", True)

        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)

        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA,
            w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_data
        )

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        x, y = self.to_screen(self.pos)
        x -= w // 2
        y -= h // 2

        glEnable(GL_TEXTURE_2D)
        glColor4f(1, 1, 1, 1)

        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(x,     y)
        glTexCoord2f(1, 1); glVertex2f(x + w, y)
        glTexCoord2f(1, 0); glVertex2f(x + w, y + h)
        glTexCoord2f(0, 0); glVertex2f(x,     y + h)
        glEnd()

        glDisable(GL_TEXTURE_2D)
        glDeleteTextures([tex_id])
    
    def _draw_pg(self, surface: pg.Surface, as_skeleton: bool = False, text: str | None = None) -> None:
        if text:
            self.text = text
        rendered_text = self.font.render(self.text, True, self.color[:3])
        text_rect = rendered_text.get_rect(center=self.to_screen(self.pos))
        surface.blit(rendered_text, text_rect)

class StarRender(ObjRender):
    def __init__(self, stars: list[Star], screen_w: int, screen_h: int, scale: int, glow: int = 10, static: bool = False, open_gl: bool = False):
        super().__init__(screen_w, screen_h, scale, static, open_gl)
        self.stars = stars
        self.glow = glow
    
    def to_screen(self, pos: ndarray, s: Star) -> list[int]:
        dx = (pos[0] - self.cam_x * s.parallax) * self.scale
        dy = (pos[1] - self.cam_y * s.parallax) * self.scale

        x = dx + self.w * 0.5
        y = self.h * 0.5 - dy
        return [int(x), int(y)]
    
    def _draw_gl(self, as_skeleton: bool = False) -> None:
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glDisable(GL_DEPTH_TEST)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

        for i in range(self.glow, 0, -1):
            glPointSize(2.0 * i)

            glBegin(GL_POINTS)
            for s in self.stars:
                x, y = self.to_screen(s.pos, s)
                if 0 <= x < self.w and 0 <= y < self.h:
                    r, g, b = [c / 255.0 for c in s.color]
                    a = 0.15 * s.luminance / i
                    glColor4f(r, g, b, a)
                    glVertex2f(x, y)
            glEnd()
        
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glPointSize(1.5)

        glBegin(GL_POINTS)
        for s in self.stars:
            x, y = self.to_screen(s.pos, s)
            if 0 <= x < self.w and 0 <= y < self.h:
                r, g, b = [c / 255.0 for c in s.color]
                a = s.luminance
                glColor4f(r, g, b, a)
                glVertex2f(x, y)
        glEnd()
    
    def _draw_pg(self, surface: pg.Surface, as_skeleton: bool = False) -> None:
        for s in self.stars:
            x, y = self.to_screen(s.pos, s)
            if 0 <= x < self.w and 0 <= y < self.h:
                r, g, b = [int(c * s.luminance) for c in s.color]
                pg.draw.circle(surface, (r, g, b), (x, y), int(s.size))