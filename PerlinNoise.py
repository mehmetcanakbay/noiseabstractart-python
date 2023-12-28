import numpy as np
from tqdm import tqdm


class PerlinNoise:
    """
    Create PerlinNoise class object.
    Specify the WIDTH AND HEIGHT, and then call the class object to create the noise.

    Parameters
    ------------
    WIDTH: int
    Specify the width of the image

    HEIGHT: int
    Specify the height of the image

    FREQ: float, default=0.005
    Frequency of the noise. Increasing this makes the noise have more "features"

    RESOLUTION: int, default=120
    Perlin Noise repeats after some time. In Perlin's original implementation, its per 256 grid point.
    For this implementation its 10 per grid point.
    WARNING: This is O(n^2) (O(n^3) in 3D), be careful when increasing it. This is also a one time operation,
    so it wont recompute the resolution everytime you call the object.

    SEED: int, default=RANDOM
    Seed of the noise.

    AMPLITUDE: int, default=1
    This is the strength of the previous layer when creating the noise.
    Explained more on OCTAVES.

    OCTAVES: int, default=6
    Perlin noise is very smooth with 1 octave. More octaves means creating a new noise by increasing the frequency,
    then adding it to the existing perlin noise with lesser strength every octave. This "strength" can be changed using
    AMPLITUDE and PERSISTENCE variables. PERSISTENCE is the decay, while AMPLITUDE is the starting strength of the noise.

    PERSISTENCE: float, default=0.7
    Used to decay the AMPLITUDE variable

    LACUNARITY: float, default=2.0
    Amount to increase the FREQ every next octave

    three_dim_enable: bool, default=False
    Enables 3D functions and disables 2D functions.

    Examples
    ------------
    2D IMAGE
    >>> noise = PerlinNoise(512,512)
    >>> image = noise()
    >>> #[-1,1] -> [0,255]
    >>> image = (image * 127.5) + 127.5
    >>> cv2.imwrite("MY_PERLIN_NOISE.png", image)

    3D IMAGE
    >>> noise = PerlinNoise(512,512, RESOLUTION=20, three_dim_enable=True)
    >>> img = np.empty((HEIGHT, WIDTH), dtype=np.float32)
    >>> for y in range(HEIGHT):
    ...     for x in range(WIDTH):
    ...         img[y,x] = noisemaker.perlin_3d_per_coord(x*0.01,y*0.01, Z_AMOUNT)
    >>> img = np.clip(img, -1.0, 1.0)
    >>> img = (img*127.5) + 127.5
    >>> img = img.astype("uint8")
    >>> cv2.imwrite("MY_3DPERLIN_NOISE.png", img)
    """

    def __init__(
        self,
        WIDTH,
        HEIGHT,
        FREQ=5e-3,
        RESOLUTION=120,
        SEED=None,
        AMPLITUDE=1,
        OCTAVES=6,
        PERSISTENCE=0.7,
        LACUNARITY=2.0,
        three_dim_enable=False,
    ):
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.FREQ = FREQ
        self.AMPLITUDE = AMPLITUDE
        self.OCTAVES = OCTAVES
        self.PERSISTENCE = PERSISTENCE
        self.LACUNARITY = LACUNARITY
        self.RESOLUTION = RESOLUTION
        self.SEED = SEED
        self.three_dim_enable = three_dim_enable
        if three_dim_enable:
            self.grid = self.make_grid_3d()
        else:
            self.grid = self.make_grid()

    # TODO: CHANGE FROM GRID BASED APPROACH TO PERMUTATION TABLE APPROACH
    # THIS IS O(n^2) with 2D noise, and O(n^3) with 3D noise. It should be changed
    def make_grid(self):
        if self.SEED != None:
            np.random.seed(self.SEED)
        grid = np.empty((self.RESOLUTION, self.RESOLUTION), dtype=np.ndarray)
        for y in range(grid.shape[1]):
            for x in range(grid.shape[0]):
                vec = np.random.normal(size=(2,))
                grid[y, x] = vec / np.linalg.norm(vec)
        return grid

    def make_grid_3d(self):
        if self.SEED != None:
            np.random.seed(self.SEED)
        grid = np.empty(
            (self.RESOLUTION, self.RESOLUTION, self.RESOLUTION), dtype=np.ndarray
        )
        for z in tqdm(range(grid.shape[2])):
            for y in range(grid.shape[1]):
                for x in range(grid.shape[0]):
                    vec = np.random.normal(size=(3,))
                    grid[z, y, x] = vec / np.linalg.norm(vec)
        return grid

    def lerp(self, x1, x2, s):
        return (x2 - x1) * ((s * (s * 6.0 - 15.0) + 10.0) * s * s * s) + x1

    def get_grid_and_dot(self, intx, inty, currx, curry):
        if self.three_dim_enable:
            raise Exception(
                "You have enabled three dimensional perlin noise. You cant access 2D functions with it enabled."
            )
        vec = self.grid[intx % self.RESOLUTION, inty % self.RESOLUTION]
        dist_vec = np.array([currx - intx, curry - inty]).reshape(
            2,
        )
        return np.dot(vec, dist_vec)

    def perlin_2d_per_coord(self, x, y):
        if self.three_dim_enable:
            raise Exception(
                "You have enabled three dimensional perlin noise. You cant access 2D functions with it enabled."
            )
        x0 = int(x // 1)
        x1 = x0 + 1
        y0 = int(y // 1)
        y1 = y0 + 1

        interp_weight_x = x - x0
        interp_weight_y = y - y0
        dot1 = self.get_grid_and_dot(x0, y0, x, y)
        dot2 = self.get_grid_and_dot(x1, y0, x, y)
        interp1 = self.lerp(dot1, dot2, interp_weight_x)

        dot1 = self.get_grid_and_dot(x0, y1, x, y)
        dot2 = self.get_grid_and_dot(x1, y1, x, y)
        interp2 = self.lerp(dot1, dot2, interp_weight_x)

        val = self.lerp(interp1, interp2, interp_weight_y)
        return val

    def get_grid_and_dot_3d(self, intx, inty, intz, currx, curry, currz):
        vec = self.grid[
            intx % self.RESOLUTION, inty % self.RESOLUTION, intz % self.RESOLUTION
        ]
        dist_vec = np.array([currx - intx, curry - inty, currz - intz]).reshape(
            3,
        )
        return np.dot(vec, dist_vec)

    def perlin_3d_per_coord(self, x, y, z):
        x0 = int(x // 1)
        x1 = x0 + 1
        y0 = int(y // 1)
        y1 = y0 + 1
        z0 = int(z // 1)
        z1 = z0 + 1

        interp_weight_x = x - x0
        interp_weight_y = y - y0
        interp_weight_z = z - z0
        dot1 = self.get_grid_and_dot_3d(x0, y0, z0, x, y, z)
        dot2 = self.get_grid_and_dot_3d(x1, y0, z0, x, y, z)
        interp1 = self.lerp(dot1, dot2, interp_weight_x)

        dot1 = self.get_grid_and_dot_3d(x0, y1, z0, x, y, z)
        dot2 = self.get_grid_and_dot_3d(x1, y1, z0, x, y, z)
        interp2 = self.lerp(dot1, dot2, interp_weight_x)

        val1 = self.lerp(interp1, interp2, interp_weight_y)

        dot1 = self.get_grid_and_dot_3d(x0, y0, z1, x, y, z)
        dot2 = self.get_grid_and_dot_3d(x1, y0, z1, x, y, z)
        interp1 = self.lerp(dot1, dot2, interp_weight_x)

        dot1 = self.get_grid_and_dot_3d(x0, y1, z1, x, y, z)
        dot2 = self.get_grid_and_dot_3d(x1, y1, z1, x, y, z)
        interp2 = self.lerp(dot1, dot2, interp_weight_x)

        val2 = self.lerp(interp1, interp2, interp_weight_y)

        val = self.lerp(val1, val2, interp_weight_z)

        return val

    def perlin_2d(self, freq):
        if self.three_dim_enable:
            raise Exception(
                "You have enabled three dimensional perlin noise. You cant access 2D functions with it enabled."
            )
        image = np.empty((self.HEIGHT, self.WIDTH), dtype=np.float32)
        for y in tqdm(range(image.shape[0])):
            for x in range(image.shape[1]):
                image[y, x] = self.perlin_2d_per_coord(x * freq, y * freq)
        return image

    def create_perlin_noise2d(self):
        if self.three_dim_enable:
            raise Exception(
                "You have enabled three dimensional perlin noise. You cant access 2D functions with it enabled."
            )
        image = np.empty((self.HEIGHT, self.WIDTH), dtype=np.float32)
        # Freq changes with every run. So create a local variable
        # to save the changes, dont change the class variable
        freq_ = self.FREQ
        amplitude_ = self.AMPLITUDE
        for _ in range(self.OCTAVES):
            tempimg = self.perlin_2d(freq_)
            image += tempimg * amplitude_
            amplitude_ *= self.PERSISTENCE
            freq_ *= self.LACUNARITY
        return image

    def __call__(self):
        return self.create_perlin_noise2d()
