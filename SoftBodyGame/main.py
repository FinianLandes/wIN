import pygame as pg
import moderngl as mgl
from physics import * 
from physics_render import *

WIDTH, HEIGHT = 1280, 720
WINDOWN_TITLE = "Test"
FPS = 60



if __name__ == "__main__":
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT))#, pg.OPENGL | pg.DOUBLEBUF)
    pg.display.set_caption(WINDOWN_TITLE)
    #ctx = mgl.create_context() # Not yet used
    clk = pg.time.Clock()
    n_points = 4
    radius = 1.0
    center = np.array([0.6, 5.0])
    points = []

    for i in range(n_points):
        angle = (2 * np.pi * i / n_points) +(np.pi / 4)
        pos = center + radius * np.array([np.cos(angle), np.sin(angle)])
        points.append(Point(pos=pos, m=1.0))
    collider = [LineCollider(point=np.array([0.0, 0.0]), normal=np.array([0.0, 1.0]), restitution=0.8)]
    body = SoftBody(points=points, collider=collider, n_sub=10, edge_alpha=0.000005, area_alpha=0.1, diag_alpha=0.0000005, damp=0.05)
    body_render = SoftBodyRender(body, WIDTH, HEIGHT, scale=50)

    running = True
    while running:
        dt = clk.tick(FPS) / 1000.0
        body.step(dt)
        screen.fill((20, 20, 40))
        body_render.draw(screen, True)
        pg.display.flip()
        
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
    pg.quit()