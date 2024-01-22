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
    "water": 3,
    "agent": 4,
}

ONE_HOT_LAYER_IDX_TO_OBJECT = dict(zip(ONE_HOT_OBJECT_TO_LAYER_IDX.values(), ONE_HOT_OBJECT_TO_LAYER_IDX.keys()))

ONE_HOT_ENCODED_OBJECTS_BY_TYPE_ID = []
for object_type in ONE_HOT_OBJECT_TO_LAYER_IDX:
    if object_type == "water":
        ONE_HOT_ENCODED_OBJECTS_BY_TYPE_ID.append(OBJECT_TO_IDX["lava"])
    else:
        ONE_HOT_ENCODED_OBJECTS_BY_TYPE_ID.append(OBJECT_TO_IDX[object_type])


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
        return self._encode_one_hot(obs, self.observation_space.shape)

    def _encode_one_hot(self, obs, out_shape):
        one_hot_encoded_obs = np.zeros(out_shape, dtype='bool')
        for position in itertools.product(range(obs.shape[0]), range(obs.shape[1])):
            object_type = self._add_encoding_for_object(obs, position, one_hot_encoded_obs)
            if object_type == "agent":
                self._add_encoding_for_underlying_object(position, one_hot_encoded_obs)
        return one_hot_encoded_obs

    def _add_encoding_for_object(self, obs, position, encoded_obs):
        object_type = self._get_object_type_at_position(*position, obs)
        layer_id = self._get_encoding_layer(object_type)
        encoded_obs[layer_id, position[0], position[1]] = 1
        return object_type

    def _add_encoding_for_underlying_object(self, position, encoded_obs):
        underlying_object_type = self._get_underlying_object_type(*position)
        additional_layer_id = self._get_encoding_layer(underlying_object_type)
        encoded_obs[additional_layer_id, position[0], position[1]] = 1

    def _get_underlying_object_type(self, i, j):
        return self.env.unwrapped.grid.get(i, j).type

    @staticmethod
    def _get_encoding_layer(object_type):
        if object_type == "lava":
            layer_id = ONE_HOT_OBJECT_TO_LAYER_IDX["water"]
        else:
            layer_id = ONE_HOT_OBJECT_TO_LAYER_IDX[object_type]
        return layer_id

    def _get_object_type_at_position(self, i, j, obs):
        object_type_id = obs[i, j]
        self._assert_that_object_can_be_encoded(object_type_id)
        object_type = IDX_TO_OBJECT[object_type_id]
        return object_type

    @staticmethod
    def _assert_that_object_can_be_encoded(object_type_id):
        assert object_type_id in ONE_HOT_ENCODED_OBJECTS_BY_TYPE_ID, \
            "The object type with idx {} is not yet " \
            "taken into account for 1-hot-encoding.".format(object_type_id)
