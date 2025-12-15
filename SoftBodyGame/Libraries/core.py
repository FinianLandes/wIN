import numpy as np
from numpy import ndarray
import math
import pygame as pg
from OpenGL.GL import *
from OpenGL.GLU import *
import time
import random
import json
from copy import deepcopy


class Point():
    def __init__(self, pos: ndarray, m: float = 1.0) -> None:
        self.pos = pos
        self.v = np.zeros_like(pos)
        self.m = m
        self.w = 1.0 / m
        self.is_grounded = False
        self.ground_normal = np.array([0.0, 1.0])
        self.game_over = False
    
    def __str__(self) -> str:
        return f"Point obj. at [{self.pos[0]:.2f}, {self.pos[1]:.2f}]"

class SlideSurface():
    def __init__(self, points: ndarray, invis: bool = False) -> None:
        self.points = points
        self.invis = invis
    def closest_point(self, point: ndarray) -> tuple[ndarray, float]: ...
    def normal_at(self, t: float) -> ndarray: ...
    def tangential_dist(self, t: float, point: ndarray) -> float: ...

class Star():
    def __init__(self, pos: ndarray, luminance: int, size: float, parallax: float, color: tuple[int] = (0, 0, 255)) -> None:
        self.pos = pos
        self.luminance = luminance
        self.size = size
        self.color = color
        self.parallax = parallax