from gymnasium import ObservationWrapper, spaces
from minigrid.core.constants import OBJECT_TO_IDX


# TODO: replace by ImgObsWrapper?
class CustomObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0,
            high=max(OBJECT_TO_IDX.values()),
            shape=(self.env.width, self.env.height),  # number of cells
            dtype="uint8",
        )

    def observation(self, observation):
        only_img = observation['image']
        only_obj_type = only_img[:, :, 0]
        return only_obj_type
