import itertools

import numpy as np
from gymnasium import ObservationWrapper, spaces
from minigrid.core.constants import OBJECT_TO_IDX, IDX_TO_OBJECT

# the object types that are occurring in the environment,
# all other object types are not considered for encoding and will raise an assertion error
ONE_HOT_OBJECT_TO_LAYER_IDX = {
    "wall": 0,
    "floor": 1,
    "goal": 2,
    "lava": 3,
    "agent": 4,
}

ONE_HOT_ENCODED_OBJECTS_BY_TYPE_ID = tuple(OBJECT_TO_IDX[object_type] for object_type in ONE_HOT_OBJECT_TO_LAYER_IDX)


def encode_one_hot(obs, out_shape):
    out = np.zeros(out_shape, dtype='bool')
    for position in itertools.product(range(obs.shape[0]), range(obs.shape[1])):
        object_type = get_object_type_at_position(*position, obs)
        layer_id = get_encoding_layer(object_type)
        out[layer_id, position[0], position[1]] = 1
    return out


def get_encoding_layer(object_type):
    layer_id = ONE_HOT_OBJECT_TO_LAYER_IDX[object_type]
    return layer_id


def get_object_type_at_position(i, j, obs):
    object_type_id = obs[i, j]
    assert_that_object_can_be_encoded(object_type_id)
    object_type = IDX_TO_OBJECT[object_type_id]
    return object_type


def assert_that_object_can_be_encoded(object_type_id):
    assert object_type_id in ONE_HOT_ENCODED_OBJECTS_BY_TYPE_ID, \
        "The object type with idx {} is not yet " \
        "taken into account for 1-hot-encoding.".format(object_type_id)


class OneHotObsWrapper(ObservationWrapper):
    """
    Wrapper to get a one-hot encoding of a fully observable
    agent view as observation.
    """

    def __init__(self, env):
        super().__init__(env)

        obs_shape = env.observation_space.shape

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(ONE_HOT_OBJECT_TO_LAYER_IDX), obs_shape[0], obs_shape[1]),
            dtype='bool'
        )

    def observation(self, obs):
        return encode_one_hot(obs, self.observation_space.shape)
