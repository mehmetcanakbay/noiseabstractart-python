import numpy as np
import random
import cv2
from common_math import rotate_in_2d
from colors import COLARR


class RandomShapes:
    """
    Create a RandomShapes class variable.
    Specify the WIDTH and HEIGHT, and then call the class object.
    It will open a openCV window, and you can choose to save the image if you press "s"
    Press "r" to re-generate the image.

    Parameters
    ------------
    WIDTH : int
    Specify the width of the image

    HEIGHT: int
    Specify the height of the image

    COLORARRAY: Tuple Array, defaults to a predefined color array if not given.
    Specify the colors that will be used when creating the images.
    Colors should be in RGB format.
    Example of an array: [(255,0,0), (255,255,255)]

    BG_COLOR: Tuple, default = (220,220,220)
    Background color of the image

    AMOUNT_OF_SHAPES: int, default=50
    Amount of shapes that will be drawn.

    ROTATION: float, default=45
    Rotation of the shapes

    SPARSITY: float, default=0.2
    Sparsity of the shapes. Should be in 0 to 1 range.

    Examples
    ------------
    >>> random_shape = RandomShapes(512,512)
    >>> random_shape()
    """

    def __init__(
        self,
        WIDTH,
        HEIGHT,
        COLORARRAY=COLARR,
        BG_COLOR=(220, 220, 220),
        AMOUNT_OF_SHAPES=50,
        ROTATION=45,
        SPARSITY=0.2,
        FORCE_TYPE=-1
    ):
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.COLORARRAY = COLORARRAY
        self.BG_COLOR = BG_COLOR
        self.AMOUNT_OF_SHAPES = AMOUNT_OF_SHAPES
        self.ROTATION = ROTATION
        self.SPARSITY = SPARSITY
        #0 for circles, 1 for square, 2 for rect
        self.FORCE_TYPE = FORCE_TYPE
        self.create_canvas()

    def create_canvas(self):
        self.image = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        # Apply BG color
        cv2.rectangle(self.image, [0, 0], [self.WIDTH, self.HEIGHT], self.BG_COLOR, -1)

    def add_coords(self, coords, amount1, amount2):
        return [coords[0] + amount1, coords[1] + amount2]

    def create_image(self):
        for _ in range(self.AMOUNT_OF_SHAPES):
            rot_amount = self.ROTATION
            randX = int(np.abs(np.random.normal(0.5, 0.2)) * self.WIDTH)
            randY = int(np.abs(np.random.normal(0.5, 0.2)) * self.HEIGHT)

            randcol = random.choice(self.COLORARRAY)[
                ::-1
            ]  # openCV's color format is BGR, not RGB. Colors are RGB, so reverse the tuple

            randW = random.randint(2, self.WIDTH // 7)
            randH = random.randint(2, self.HEIGHT // 6)
            if self.FORCE_TYPE == -1:
                rnd_shape = random.randint(0, 2)
            if self.FORCE_TYPE != -1:
                rnd_shape = self.FORCE_TYPE
            if rnd_shape == 0:
                # Circle
                cv2.circle(
                    self.image,
                    (int(randX), int(randY)),
                    int(randW) // 2,
                    randcol,
                    -1,
                    lineType=cv2.LINE_AA,
                )
            if rnd_shape == 1:
                # Square
                ##This whole thing is for rotation.
                # Basically I save the randX variables, then subtract them so I put the random coords at 0,0. The origin.
                # Rotation matrix rotates these objects from the origin. Pivot is the origin basically.
                # After the rotation I add the location variables back.
                randXsave = int(randX)
                randYsave = int(randY)
                randX -= randX
                randY -= randY
                coords1 = rotate_in_2d([randX, randY], rot_amount)
                coords2 = rotate_in_2d([randX, randY + randW], rot_amount)
                coords3 = rotate_in_2d([randX + randW, randY + randW], rot_amount)
                coords4 = rotate_in_2d([randX + randW, randY], rot_amount)
                coords = np.array(
                    [
                        self.add_coords(coords1, randXsave, randYsave),
                        self.add_coords(coords2, randXsave, randYsave),
                        self.add_coords(coords3, randXsave, randYsave),
                        self.add_coords(coords4, randXsave, randYsave),
                    ]
                )
                #######
                cv2.fillPoly(self.image, [coords], randcol, lineType=cv2.LINE_AA)
            if rnd_shape == 2:
                # Rectangle
                # Not going to reformat this to a function because coords are computed differently, randH is added in rect
                randXsave = int(randX)
                randYsave = int(randY)
                randX -= randX
                randY -= randY
                coords1 = rotate_in_2d([randX, randY], rot_amount)
                coords2 = rotate_in_2d([randX, randY + randH], rot_amount)
                coords3 = rotate_in_2d([randX + randW, randY + randH], rot_amount)
                coords4 = rotate_in_2d([randX + randW, randY], rot_amount)
                coords = np.array(
                    [
                        self.add_coords(coords1, randXsave, randYsave),
                        self.add_coords(coords2, randXsave, randYsave),
                        self.add_coords(coords3, randXsave, randYsave),
                        self.add_coords(coords4, randXsave, randYsave),
                    ]
                )
                cv2.fillPoly(self.image, [coords], randcol, lineType=cv2.LINE_AA)

    def show_image(self):
        cv2.imshow("RandomShapes", self.image)
        key = cv2.waitKey(0)
        if key == ord("s"):
            cv2.imwrite("RandomShape.png", self.image)
        if key == ord("r"):
            self.create_canvas()
            self.create_image()
            self.show_image()

    def __call__(self):
        self.create_image()
        self.show_image()

# mycollarray = [(i,i,i) for i in range(255)]
mycollarray = [(222,24,74), (164,67,91), (146,16,48), (194,79,108),(94,10,31)]
rs = RandomShapes(1024, 1024, AMOUNT_OF_SHAPES=200, FORCE_TYPE=-1, COLORARRAY=mycollarray)
rs()
