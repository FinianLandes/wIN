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
        self.track_generator: TrackGenerator | None = None
        self.surfaces: list[SlideSurface] = []
        self.colliders: list[Collider] = []
        self.player_points = [Point([-0.2, 12.0]), Point([0.2, 12.0]), Point([0.2, 12.4]), Point([-0.2, 12.4])]
        self.player: ShapedSoftBody | None = None
        self.soft_bodies: list[ShapedSoftBody] = []
        self.renders: list[ObjRender] = []
        self.skeleton_view = False
        self.jump_impulse = 13
        self.surface_restitution = 0.99
        self.game_over = False
        self.last_player_pos: ndarray | None = None
        self.score = 0
        self.high_score = 0
        self.game_over_time = 0
        self.game_over_timeout = 1.0
        self.start_again = False
        self.font = "Comic Sans MS"
        self.screen, self.clk = self._setup()
        self._start_game()

    def run(self) -> None:
        while self._is_running():
            dt = self.clk.tick(self.fps) / 1000.0
            self._update_physics(dt)
            self._update_colliders()
            self._update_game()
            self._update_graphics()
            self._handle_events()
        self._quit()
    
    def _is_running(self) -> bool:
        return self.running
    
    def _update_physics(self, dt: float) -> None:
        for b in self.soft_bodies:
            b.step(dt)
            self.game_over = any(p.game_over for p in b.points)
        if not self.game_over:
            self.score = np.sqrt((self.player_points[0].pos[0] - self.player.centroid[0]) ** 2 + (self.player_points[0].pos[1] - self.player.centroid[0]) ** 2) * 10 / self.scale 
    
    def _update_colliders(self) -> None:
        self.track_generator.update(self.player.centroid)
        self.surfaces = self.track_generator.segments
        self.colliders = [Collider(s, restitution=self.surface_restitution, invis=s.invis) for s in self.surfaces]
        for b in self.soft_bodies:
            b.colliders = self.colliders

    def _update_graphics(self) -> None:
        text_render = []
        if not self.game_over:
            self.last_player_pos = self.player.centroid
            text_render += [TextRender(f"Score: {int(self.score)}", self.width, self.height, self.scale, pg.font.SysFont(self.font, 30), pos=np.array([7.4, 5.6]), static=True)]
            self.renders = [SoftBodyRender(b, self.width, self.height, self.scale) for b in self.soft_bodies] + [SlideSurfaceRender(s,self.width, self.height, self.scale) for s in self.surfaces if not s.invis] + text_render
        if self.game_over and time.time() - self.game_over_time > self.game_over_timeout: 
            text_render += [TextRender(f"Highscore: {int(self.high_score)}", self.width, self.height, self.scale, pg.font.SysFont(self.font, 30), pos=np.array([0.0, -1.0]), static=True)]
            text_render += [TextRender(f"Game Over", self.width, self.height, self.scale, pg.font.SysFont(self.font, 50), pos=np.array([0.0, 0.0]), static=True)]
            text_render += [TextRender(f"Press Space to continue", self.width, self.height, self.scale, pg.font.SysFont(self.font, 30), pos=np.array([0.0, -3.0]), static=True)]
            self.renders = text_render
            
        self.screen.fill((20, 20, 40))
        for r in self.renders:
            r.update_camera(self.last_player_pos)
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
            if self.game_over and time.time() - self.game_over_time > self.game_over_timeout: 
                self.start_again = True
                return
            for b in self.soft_bodies:
                ground_normal = np.array([0.0, 1.0])
                has_ground = False
                for p in b.points:
                    if p.is_grounded and np.linalg.norm(p.ground_normal) > 0.1:
                        ground_normal += p.ground_normal
                        has_ground = True
                if has_ground:
                    ground_normal = ground_normal / np.linalg.norm(ground_normal)
                ground_normal[0] += 0.2
                if b.can_jump():
                    b.add_impulse(self.jump_impulse * ground_normal)
        if key == pg.K_RIGHT:
            for b in self.soft_bodies:
                ground_normal = np.array([0.0, 1.0])
                has_ground = False
                for p in b.points:
                    if p.is_grounded and np.linalg.norm(p.ground_normal) > 0.1:
                        ground_normal += p.ground_normal
                        has_ground = True

                if has_ground:
                    ground_normal = ground_normal / np.linalg.norm(ground_normal)

                tan = np.array([ground_normal[1], -ground_normal[0]])
                if b.can_jump():
                    b.add_impulse(self.jump_impulse * tan)

    def _update_game(self):
        if self.game_over:
            self.high_score = self.score if self.score > self.high_score else self.high_score
            if self.game_over_time == 0:
                self.game_over_time = time.time()
        if self.start_again:
            self.start_again = False
            self.game_over = False
            self.game_over_time = 0
            self.score = 0.0
            self._start_game()
        
    def _start_game(self) -> None:
        self.track_generator = TrackGenerator()
        self.colliders = []
        self.surfaces = []
        self.player = ShapedSoftBody(deepcopy(self.player_points), self.colliders, damp=0.02, stiffness=0.3)
        self.soft_bodies = [self.player]
        self.last_player_pos = np.array([0.0, 0.0])

    def _setup(self) -> tuple[pg.Surface, pg.time.Clock]:
        pg.init()
        pg.font.init()
        screen = pg.display.set_mode((self.width, self.height))#, pg.OPENGL | pg.DOUBLEBUF)
        pg.display.set_caption(self.window_title)
        clk = pg.time.Clock()
        return screen, clk
    
    def _quit(self) -> None:
        pg.quit()
    