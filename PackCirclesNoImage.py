import numpy as np
import cv2
import random
from common_math import dist

class PackCirclesImage:
    """
    Create a RandomCircles class variable.
    Specify the WIDTH and HEIGHT, and then call the class object.
    It will open a openCV window, and you can choose to save the image if you press "s"

    Parameters
    ------------
    WIDTH : int
    Specify the width of the image

    HEIGHT: int
    Specify the height of the image

    IMAGE_PATH: string
    Image location, with image name and extension.

    MIN_RADIUS: int, default=4
    Specify the minimum radius of the circles

    MAX_RADIUS: int, default=FLOOR(WIDTH/6)
    Specify the maximum radius of the circles

    SPACE_BETWEEN_CIRCLES: float, default=0.5
    How much pixels should be left between two circles

    AMOUNT: int, default=500
    Amount of circles that will be drawn.

    BG_COLOR: Tuple, default = (220,220,220)
    Background color of the image

    Examples
    ------------
    >>> rc = PackCirclesImage(512, 512, "images/test_image.jpg")
    >>> rc()
    """

    def __init__(
        self,
        WIDTH,
        HEIGHT,
        COLOR,
        MIN_RADIUS=4,
        MAX_RADIUS=None,
        SPACE_BETWEEN_CIRCLES=0.5,
        AMOUNT=500,
        BGCOLOR=(0, 0, 0),
    ):
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.COLOR = COLOR
        self.REF_IMAGE = None
        self.AMOUNT = AMOUNT
        if MAX_RADIUS == None:
            self.MAX_RADIUS = WIDTH // 45
        else:
            self.MAX_RADIUS = MAX_RADIUS
        self.MIN_RADIUS = MIN_RADIUS
        self.SPACE_BETWEEN_CIRCLES = SPACE_BETWEEN_CIRCLES
        self.image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        cv2.rectangle(self.image, [0, 0], [WIDTH, HEIGHT], BGCOLOR, -1)
        self.circles = []


    def find_circles(self):
        iterations = 0
        print(f"{len(self.circles)} circles\r", end=" ", flush=True)
        while len(self.circles) != self.AMOUNT:  # SHAPE AMOUNT
            minRadius = self.MIN_RADIUS
            maxRadius = self.MAX_RADIUS
            circle = {
                "x": random.randint(0, self.WIDTH),
                "y": random.randint(0, self.HEIGHT),
                "r": random.randint(minRadius, maxRadius),
            }

            mindist = 0xFFFF
            for j in range(len(self.circles)):
                other = self.circles[j]
                currdistance = (
                    dist([circle["x"], circle["y"]], [other["x"], other["y"]])
                    - other["r"]
                    - self.SPACE_BETWEEN_CIRCLES
                )
                mindist = min(currdistance, mindist)

            if mindist > minRadius:
                try:
                    circle["r"] = random.randint(
                        minRadius, min(maxRadius, int(mindist))
                    )
                except ValueError:  # if its equal random.randint returns ValueError
                    circle["r"] = minRadius

                self.circles.append(circle)

            if iterations % 1500 == 0:
                print(
                    f"{len(self.circles)} circles, {iterations} iterations"
                    + (". " * (iterations // 1500)),
                    end="\r ",
                    flush=True,
                )
            iterations += 1

    def draw_circles(self):
        for circle in self.circles:
            # col = self.REF_IMAGE[circle["y"], circle["x"], :]
            # col = (int(col[0]), int(col[1]), int(col[2]))
            col = self.COLOR
            cv2.circle(
                self.image,
                (circle["x"], circle["y"]),
                circle["r"],
                col,
                -1,
                lineType=cv2.LINE_AA,
            )

    def show_image(self):
        cv2.imshow("PackCirclesNoImage", self.image)
        key = cv2.waitKey(0)
        if key == ord("s"):
            cv2.imwrite("PackCirclesNoImage.png", self.image)

    def __call__(self):
        self.find_circles()
        self.draw_circles()
        self.show_image()


rc = PackCirclesImage(1280, 720, (255,255,255), MAX_RADIUS=10, AMOUNT=3000, SPACE_BETWEEN_CIRCLES=0.2)
rc()
