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
        self.track_generator = TrackGenerator()
        self.surfaces: list[SlideSurface] = []
        self.colliders: list[Collider] = [Collider(s, restitution=0.95) for s in self.surfaces]

        self.player_points = [Point([-0.2, 12.0]), Point([0.2, 12.0]), Point([0.2, 12.4]), Point([-0.2, 12.4])]
        self.player = ShapedSoftBody(self.player_points,self.colliders, damp=0.05)
        self.soft_bodies: list[ShapedSoftBody] = [self.player]
        self.renders: list[ObjRender] = []
        self.skeleton_view = True
        self.jump_impulse = np.array([0.0, 10.0])
        self.surface_restitution = 0.95
        self.screen, self.clk = self._setup()

    def run(self) -> None:
        while self._is_running():
            dt = self.clk.tick(self.fps) / 1000.0
            self._update_physics(dt)
            self._update_colliders()
            self._update_graphics()
            self._handle_events()
        self._quit()
    
    def _is_running(self) -> bool:
        return self.running
    
    def _update_physics(self, dt: float) -> None:
        for b in self.soft_bodies:
            b.step(dt)
    
    def _update_colliders(self) -> None:
        self.track_generator.update(self.player.centroid)
        self.surfaces = self.track_generator.segments
        self.colliders = [Collider(s, restitution=self.surface_restitution) for s in self.surfaces]
        for b in self.soft_bodies:
            b.colliders = self.colliders


    def _update_graphics(self) -> None:
        self.renders = [SoftBodyRender(b, self.width, self.height, self.scale) for b in self.soft_bodies] + [SlideSurfaceRender(s,self.width, self.height, self.scale) for s in self.surfaces]
        self.screen.fill((20, 20, 40))
        for r in self.renders:
            r.update_camera(self.player.centroid)
            r.draw(self.screen, self.skeleton_view)
        pg.display.flip()
    
    def _handle_events(self) -> None:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            if event.type == pg.KEYDOWN:
                self._handle_keydown(event.key)
    
    def _handle_keydown(self, key: int) -> None:
        if key == pg.K_SPACE:
            for b in self.soft_bodies:
                if b.can_jump():
                    b.add_impulse(self.jump_impulse)
        if key == pg.K_RIGHT:
            for b in self.soft_bodies:
                b.add_impulse(np.array([10, 0]))

    
    def _quit(self) -> None:
        pg.quit()
    
    def _setup(self) -> tuple[pg.Surface, pg.time.Clock]:
        pg.init()
        screen = pg.display.set_mode((self.width, self.height))#, pg.OPENGL | pg.DOUBLEBUF)
        pg.display.set_caption(self.window_title)
        clk = pg.time.Clock()
        return screen, clk