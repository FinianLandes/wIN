import numpy as np
from numpy import ndarray
from physics import *
import pygame as pg

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
    def __init__(self, body: SoftBody, screen_w: int, screen_h: int, scale: float = 100.0, color_rgba: tuple[int] = (255, 200, 100, 80)) -> None:
        super().__init__(screen_w, screen_h, scale)
        self.body = body
        self.color = color_rgba
        self.thickness = 4
    
    def draw(self, surface: pg.Surface, as_skeleton: bool = False) -> None:
        pts = [self.to_screen(p.pos) for p in self.body.points]
        n = len(pts)
        if not as_skeleton:
            pg.draw.polygon(surface=surface, color=self.color, points=pts)
            for i in range(n):
                pg.draw.line(surface, self.color[:3], pts[i], pts[(i+1)%n], self.thickness)
        else:
            for c in self.body.C:
                if isinstance(c, DistConstraint):
                    p0 = self.to_screen(c.points[0].pos)
                    p1 = self.to_screen(c.points[1].pos)
                    pg.draw.line(surface, self.color[:3], p0, p1, 2)

            for p in pts:
                pg.draw.circle(surface, self.color[:3], p, 5)
