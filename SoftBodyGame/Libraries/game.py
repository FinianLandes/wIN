from .core import *
from .physics import * 
from .render import *
from .objects import *

class Game:
    def __init__(self, screen_size: tuple[int], scale: int, fps: int, title: str = "Softbody Game", open_gl: bool = True) -> None:
        self.width, self.height = screen_size
        self.scale = scale
        self.true_width = (self.width / self.scale, self.height / self.scale)
        self.fps = fps
        self.window_title = title
        self.font = "Comic Sans MS"
        self.open_gl = open_gl
        self.bg_color = (20, 20, 40)

        self.game_over = False
        self.start_again = False
        self.startup = True
        self.skeleton_view = False

        self.score = 0.0
        self.high_score = 0.0
        self.game_over_time = 0.0
        self.game_over_timeout = 1.0

        self.jump_impulse = 8.0 if self.open_gl else 10.0
        self.surface_restitution = 0.99

        self.track_generator: TrackGenerator | None = None
        self.bg_gen: BgGenerator | None = None

        self.surfaces: list[SlideSurface] = []
        self.colliders: list[Collider] = []

        self.player_points = self._load_body()
        self.player: ShapedSoftBody | None = None
        self.soft_bodies: list[ShapedSoftBody] = []

        self.star_render: list[StarRender] = []
        self.soft_body_renders: list[SoftBodyRender] = []
        self.collide_render: list[SlideSurfaceRender] = []
        self.game_over_renders: list[TextRender] = []
        self.renders: list[ObjRender] = []

        self.last_player_pos: ndarray | None = None

        self.screen, self.clk = self._setup()
        self.running = True
        self._start_game()

    def run(self) -> None:
        while self._is_running():
            dt = self.clk.tick(self.fps) / 1000.0
            self._update_physics(dt)
            self._update_elements()
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
    
    def _update_elements(self) -> None:
        if self.bg_gen.update(self.player.centroid):
            self.star_render = [StarRender(self.bg_gen.get_stars(), self.width, self.height, self.scale, open_gl=self.open_gl)]
        
        if self.track_generator.update(self.player.centroid):
            self.surfaces = self.track_generator.segments
            self.colliders = [Collider(s, restitution=self.surface_restitution, invis=s.invis) for s in self.surfaces]
            self.collide_render = [SlideSurfaceRender(s,self.width, self.height, self.scale, glow=10, open_gl=self.open_gl) for s in self.surfaces if not s.invis] 
        
        for b in self.soft_bodies:
            b.colliders = self.colliders

    def _update_graphics(self) -> None:
        text_render = []
        if not self.game_over:
            self.last_player_pos = self.player.centroid
            text_render += [TextRender(f"Score: {int(self.score)}", self.width, self.height, self.scale, pg.font.SysFont(self.font, 30), pos=np.array([7.4, 5.6]),glow=0, static=True, open_gl=self.open_gl)]
            if self.startup:
                text_render += [TextRender(f"Press Space to Jump", self.width, self.height, self.scale, pg.font.SysFont(self.font, 30), pos=np.array([1.0, 5.0]),glow=0, static=False, open_gl=self.open_gl)]
            
            self.renders = self.soft_body_renders + self.collide_render + text_render + self.star_render
        
        if self.game_over and time.time() - self.game_over_time > self.game_over_timeout: 
            self.game_over_renders[0].text = f"Highscore: {int(self.high_score)}"
            self.renders = self.game_over_renders
            
        if self.open_gl:
            glClearColor(self.bg_color[0]/255, self.bg_color[1]/255, self.bg_color[2]/255, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
        else:
            self.screen.fill(self.bg_color)

        for r in self.renders:
            if not r.static:
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
            self.startup = False
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
        self.bg_gen = BgGenerator()
        self.player = ShapedSoftBody(deepcopy(self.player_points), self.colliders, damp=0.05, stiffness=0.5)
        self.soft_bodies = [self.player]
        self.soft_body_renders = [SoftBodyRender(b, self.width, self.height, self.scale, glow=3, open_gl=self.open_gl) for b in self.soft_bodies]
        self.last_player_pos = np.array([0.0, 0.0])

    def _setup(self) -> tuple[pg.Surface, pg.time.Clock]:
        pg.init()
        pg.font.init()
        if self.open_gl:
            screen = pg.display.set_mode((self.width, self.height), pg.OPENGL | pg.DOUBLEBUF)
            glViewport(0, 0, self.width, self.height)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluOrtho2D(0, self.width, self.height, 0)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDisable(GL_DEPTH_TEST)
        else:
            screen = pg.display.set_mode((self.width, self.height))
        pg.display.set_caption(self.window_title)

        self.game_over_renders = [TextRender("", self.width, self.height, self.scale, pg.font.SysFont(self.font, 30), pos=np.array([0.0, -1.0]), glow=0, static=True, open_gl=self.open_gl), TextRender(f"Game Over", self.width, self.height, self.scale, pg.font.SysFont(self.font, 50), pos=np.array([0.0, 0.0]), glow=0, static=True, open_gl=self.open_gl), TextRender(f"Press Space to continue", self.width, self.height, self.scale, pg.font.SysFont(self.font, 30), pos=np.array([0.0, -3.0]), glow=0, static=True, open_gl=self.open_gl)]

        clk = pg.time.Clock()
        return screen, clk
    
    def _load_body(self) -> list[Point]:
        with open("SoftBodyGame/Resources/Seal.json") as f:
            data = json.load(f)

        points_array = np.array(data["points"], dtype=float)

        scale = 0.4
        points_array *= scale
        lst = []
        for p in points_array:
            p[1] += 6.0
            lst.append(Point(p))

        return lst
    
    def _quit(self) -> None:
        pg.quit()
