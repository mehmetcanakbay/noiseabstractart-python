import random
import numpy as np
import pygame
from PerlinNoise import PerlinNoise


class Particle:
    def __init__(
        self,
        width,
        height,
        freq,
        strength_of_field,
        particle_speed,
        linewidth=1,
        color=(64, 64, 64, 10),
        pos=None,
    ):
        if pos == None:
            self.pos = [random.randint(0, width), random.randint(0, height), 0]
        else:
            self.pos = pos
        self.width = width
        self.height = height
        self.freq = freq
        self.strength_of_field = strength_of_field
        self.particle_speed = particle_speed
        self.linewidth = linewidth
        self.vel = [0, 0, 0]
        self.acc = [0, 0, 0]
        self.color = color
        self.last_pos = self.pos
        self.should_be_destroyed = False

    def update(self):
        self.vel = self.clamp_vec(self.vel, 0.0, self.particle_speed)
        self.vel = [self.vel[i] + self.acc[i] for i in range(3)]
        self.last_pos = self.pos
        self.pos = [self.pos[i] + self.vel[i] for i in range(3)]
        self.acc = [self.acc[i] * 0 for i in range(3)]

    def apply_force(self, force):
        self.acc = [self.acc[i] + force[i] for i in range(3)]

    def edges(self):
        if self.pos[0] > self.width:
            self.should_be_destroyed = True
        if self.pos[0] < 0:
            self.should_be_destroyed = True
        if self.pos[1] > self.height:
            self.should_be_destroyed = True
        if self.pos[1] < 0:
            self.should_be_destroyed = True

    def show(self, window):
        # if i dont always make a new screen it loses its "transparency"
        # disappointing but dont have a choice, huge performance impact
        surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for i in range(self.linewidth):
            pygame.draw.aaline(
                surf,
                self.color,
                (self.last_pos[0] + i, self.last_pos[1]),
                (self.pos[0] + i, self.pos[1]),
            )
            pygame.draw.aaline(
                surf,
                self.color,
                (self.last_pos[0] - i, self.last_pos[1]),
                (self.pos[0] - i, self.pos[1]),
            )
            pygame.draw.aaline(
                surf,
                self.color,
                (self.last_pos[0], self.last_pos[1] + i),
                (self.pos[0], self.pos[1] + i),
            )
            pygame.draw.aaline(
                surf,
                self.color,
                (self.last_pos[0], self.last_pos[1] - i),
                (self.pos[0], self.pos[1] - i),
            )
        window.blit(surf, (0, 0))

    def follow(self, noise, z):
        x = int(np.floor(self.pos[0] - 1))
        y = int(np.floor(self.pos[1] - 1))
        angl = noise.perlin_3d_per_coord(x * self.freq, y * self.freq, z) * np.pi * 2
        # angle to 2d vector
        vec = [np.sin(angl), np.cos(angl)]
        vec = self.scale_2d_vector(vec, self.strength_of_field)
        # vec2 -> vec3 through +[0]
        self.apply_force(vec + [0])

    def scale_2d_vector(self, vec, scale_fac):
        vec = vec / (np.sqrt((vec[0] ** 2 + vec[1] ** 2)))
        vec = [vec[i] * scale_fac for i in range(2)]
        return vec

    def clamp_vec(self, vec, low, high):
        for i in range(len(vec)):
            if vec[i] > high:
                vec[i] = high
            if vec[i] < low:
                vec[i] = low
        return vec


class Flowfield:
    def __init__(
        self,
        width: int,
        height: int,
        bgcolor: tuple,
        freq: float,
        stepamount_in_three_dim: float,
        particle_amount: int,
        particle_speed: float,
        strength_of_field: float,
        color=(64, 64, 64, 10),
        linewidth=1,
        noise_resolution=15,
        show_grid=False,
    ):
        self.width = width
        self.height = height
        self.bgcolor = bgcolor
        self.freq = freq
        self.particle_amount = particle_amount
        self.particle_speed = particle_speed
        self.strength_of_field = strength_of_field
        self.noise_resolution = noise_resolution
        self.linewidth = linewidth
        self.color = color
        self.show_grid = show_grid
        self.stepamount_in_three_dim = stepamount_in_three_dim
        self.particles = [
            Particle(
                width, height, freq, strength_of_field, particle_speed, linewidth, color
            )
            for x in range(self.particle_amount)
        ]
        self.noise = PerlinNoise(
            128, 128, three_dim_enable=True, RESOLUTION=self.noise_resolution
        )
        self.screenshot_amount = 0

    def initialize_pygame_window(self):
        pygame.init()
        self.window = pygame.display.set_mode((self.width, self.height))
        self.window.fill(self.bgcolor)

    def show_screen(self):
        z = 0
        while True:
            # show grid is basically "debug" mode.
            # have to just fill the window, this basically wipes everything about particles.
            # Couldnt find a solution, it seems to be a limitation of pygame. Not going to dig deeper.
            # therefore its not needed to calculate all the particle location.
            if self.show_grid:
                self.window.fill((255, 255, 255, 255))
                for y in range(0, self.height, 20):
                    for x in range(0, self.width, 20):
                        angl = (
                            self.noise.perlin_3d_per_coord(
                                x * self.freq, y * self.freq, z
                            )
                            * np.pi
                            * 2
                        )
                        # *10 are there for visualization, otherwise the lines look like dots
                        pygame.draw.line(
                            self.window,
                            (255, 0, 0),
                            (x, y),
                            (x + int(np.cos(angl) * 10), y + int(np.sin(angl) * 10)),
                        )
            else:
                for i, particle in enumerate(self.particles):
                    if particle.should_be_destroyed:
                        # Make a new particle instead of it, effectively killing it
                        self.particles[i] = Particle(
                            self.width,
                            self.height,
                            self.freq,
                            self.strength_of_field,
                            self.particle_speed,
                            self.linewidth,
                            self.color,
                        )
                    particle.follow(self.noise, z)
                    # pos update
                    particle.update()
                    # if touched edge, should be destroyed flag on
                    particle.edges()
                    particle.show(self.window)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_k:
                        pygame.image.save(
                            self.window, f"flowfield_screenshot{self.screenshot_amount}.png"
                        )
                        self.screenshot_amount += 1
            pygame.display.update()
            z += self.stepamount_in_three_dim

"""
field = Flowfield(
    512, 512, (220, 220, 220), 0.005, 0.008, 100, 15, 10, color=(128, 30, 10, 10)
)
field.initialize_pygame_window()
field.show_screen()
"""