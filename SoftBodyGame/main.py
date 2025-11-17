import pygame as pg
import moderngl as mgl
from Libraries.physics import * 
from Libraries.render import *
from Libraries.objects import *

WIDTH, HEIGHT = 1280, 720
SCALE = 30
TRUE_WIDTH, TRUE_HEIGHT = WIDTH / SCALE, HEIGHT / SCALE
WINDOWN_TITLE = "Test"
FPS = 60




if __name__ == "__main__":
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT))#, pg.OPENGL | pg.DOUBLEBUF)
    pg.display.set_caption(WINDOWN_TITLE)
    #ctx = mgl.create_context() # Not yet used
    clk = pg.time.Clock()

    n_points = 40
    radius = 1.0
    center = np.array([0.0, 9.0])
    body_points = []

    for i in range(n_points):
        angle = (2 * np.pi * i / n_points) +(np.pi / 4)
        pos = center + radius * np.array([np.cos(angle), np.sin(angle)])
        body_points.append(Point(pos=pos, m=1.0))

    line_points = np.array([[5, 10], [3, 4]])

    bezier_points = np.array([[5, 5], [2, 3], [3, 4]])
    
    surfaces = [LineSlideSurface(np.array([[-TRUE_WIDTH, 0], [TRUE_WIDTH, 0]])), LineSlideSurface(line_points)]
    colliders = [Collider(s, restitution=0.95) for s in surfaces]
    soft_bodies = [ShapedSoftBody(points=body_points, colliders=colliders, damp=0.01, rot_damp=0.08, stiffness=0.5)]
    renders = [SoftBodyRender(body, WIDTH, HEIGHT, scale=SCALE) for body in soft_bodies] + [SlideSurfaceRender(s, WIDTH, HEIGHT, scale=SCALE) for s in surfaces]


    skeleton_view: bool = False
    running = True
    while running:
        dt = clk.tick(FPS) / 4000.0
        for b in soft_bodies:
            b.step(dt)

        screen.fill((20, 20, 40))

        for r in renders:
            r.draw(screen, skeleton_view)
        pg.display.flip()

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
    pg.quit()