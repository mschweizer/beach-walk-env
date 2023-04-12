from gym_minigrid.minigrid import Lava
from gym_minigrid.rendering import fill_coords, point_in_rect, point_in_line


class Water(Lava):

    def render(self, img):
        # Background color for water
        blue_color = (0, 0, 255)
        fill_coords(img, point_in_rect(0, 1, 0, 1), blue_color)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))