from .core import *
from .physics import * 
from .render import *
from .objects import *

class Game:
    def __init__(self, screen_size: tuple[int], scale: int, fps: int, title: str = "Game") -> None:
        self.width, self.height = screen_size
        self.scale = scale
        self.true_width = self.width / self.scale, self.height / self.scale
        self.window_title = title
        self.fps = fps
        self.running = True
        self.surfaces: list[SlideSurface] = []
        self.colliders: list[Collider] = [Collider(s, restitution=0.95) for s in self.surfaces]
        self.player = SoftBody()
        self.soft_bodies: list[SoftBody] = [self.player]
        self.renders: list[ObjRender] = []
        self.skeleton_view = False
        self.jump_impulse = np.array([0.0, 10.0])
        self.screen, self.clk = self._setup()

    def run(self) -> None:
        while self._is_running():
            dt = self.clk.tick(self.fps) / 1000.0
            self._update_physics(dt)
            self._update_graphics()
            self._handle_events()
        self._quit()
    
    def _is_running(self) -> bool:
        return self.running
    
    def _update_physics(self, dt: float) -> None:
        for b in self.soft_bodies:
            b.step(dt)
    
    def _update_graphics(self) -> None:
        self.screen.fill((20, 20, 40))
        for r in self.renders:
            r.draw(self.screen, self.skeleton_view)
        pg.display.flip()
    
    def _handle_events(self) -> None:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            if event.type == pg.KEYDOWN:
                self._handle_keydown(event.key)
    
    def _handle_keydown(self, key: int) -> None:
        for b in self.soft_bodies:
            if b.can_jump():
                b.add_impulse(self.jump_impulse)
    
    def _quit(self) -> None:
        pg.quit()
    
    def _setup(self) -> tuple[pg.Surface, pg.time.Clock]:
        pg.init()
        screen = pg.display.set_mode((self.width, self.height))#, pg.OPENGL | pg.DOUBLEBUF)
        pg.display.set_caption(self.window_title)
        clk = pg.time.Clock()
        return screen, clk

"""
WIDTH, HEIGHT = 1280, 720
SCALE = 30
TRUE_WIDTH, TRUE_HEIGHT = WIDTH / SCALE, HEIGHT / SCALE
WINDOWN_TITLE = "Test"
FPS = 60
JUMP_IMPULSE = np.array([0.0, 10.0])


if __name__ == "__main__":
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT))#, pg.OPENGL | pg.DOUBLEBUF)
    pg.display.set_caption(WINDOWN_TITLE)
    #ctx = mgl.create_context() #Currently unused
    

    n_points = 50
    radius = 1.0
    center = np.array([17, 11.0])
    body_points = []

    for i in range(n_points):
        angle = (2 * np.pi * i / n_points) +(np.pi / 4)
        pos = center + radius * np.array([np.cos(angle), np.sin(angle)])
        body_points.append(Point(pos=pos, m=1.0))

    line_points = np.array([[5, 0.5], [12.5, 2.5]], dtype=float)

    bezier_points = np.array([[20, 10], [21, 3], [13, 2.99]], dtype=float)

    
    surfaces = [LineSlideSurface(np.array([[0,0], [TRUE_WIDTH, 0]])), BezierSlideSurface(bezier_points, cw=True), LineSlideSurface(line_points)]
    colliders = [Collider(s, restitution=0.95) for s in surfaces]
    soft_bodies = [ShapedSoftBody(points=body_points, colliders=colliders, damp=0.01, rot_damp=0.08, stiffness=0.8)]
    renders = [SoftBodyRender(body, WIDTH, HEIGHT, scale=SCALE) for body in soft_bodies] + [SlideSurfaceRender(s, WIDTH, HEIGHT, scale=SCALE) for s in surfaces]
    #text = TextRender("Can Jump: ", WIDTH, HEIGHT, SCALE, font=pg.font.SysFont('Arial', 30, bold=False), pos=np.array([1.0, TRUE_HEIGHT - 2]))


    skeleton_view: bool = False
    running = True
    while running:
        dt = clk.tick(FPS) / 1000.0
        for b in soft_bodies:
            b.step(dt)

        screen.fill((20, 20, 40))

        for r in renders:
            r.draw(screen, skeleton_view)
        
        pg.display.flip()

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    
    pg.quit()
"""