import numpy as np
from numpy import ndarray
import math
import pygame as pg
import moderngl as mgl
import time
import random


class Point():
    def __init__(self, pos: ndarray, m: float = 1.0) -> None:
        self.pos = pos
        self.v = np.zeros_like(pos)
        self.m = m
        self.w = 1.0 / m
        self.is_grounded = False

class SlideSurface():
    def __init__(self, points: ndarray) -> None:
        self.points = points
    def closest_point(self, point: ndarray) -> tuple[ndarray, float]: ...
    def normal_at(self, t: float) -> ndarray: ...

