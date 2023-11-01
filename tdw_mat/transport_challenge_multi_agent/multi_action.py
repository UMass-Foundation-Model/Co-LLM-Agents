from abc import ABC
from typing import List, Optional
from overrides import final
from tdw.replicant.replicant_static import ReplicantStatic
from tdw.replicant.replicant_dynamic import ReplicantDynamic
from tdw.replicant.image_frequency import ImageFrequency
from tdw.replicant.actions.action import Action
from transport_challenge_multi_agent.challenge_state import ChallengeState


class MultiAction(Action, ABC):
    """
    Abstract base class for actions that are divided into multiple "sub-actions".
    """

    def __init__(self, state: ChallengeState):
        """
        :param state: The [`ChallengeState`](challenge_state.md) data.
        """

        super().__init__()
        self._state: ChallengeState = state
        self._sub_action: Optional[Action] = None
        self._image_frequency: ImageFrequency = ImageFrequency.once

    def get_initialization_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic,
                                    image_frequency: ImageFrequency) -> List[dict]:
        """
        :param resp: The response from the build.
        :param static: The [`ReplicantStatic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_static.md) data that doesn't change after the Replicant is initialized.
        :param dynamic: The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call.
        :param image_frequency: An [`ImageFrequency`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/image_frequency.md) value describing how often image data will be captured.

        :return: A list of commands to initialize this action.
        """

        self._image_frequency = image_frequency
        return super().get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                   image_frequency=image_frequency)

    @final
    def get_end_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic,
                         image_frequency: ImageFrequency) -> List[dict]:
        """
        :param resp: The response from the build.
        :param static: The [`ReplicantStatic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_static.md) data that doesn't change after the Replicant is initialized.
        :param dynamic: The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call.
        :param image_frequency: An [`ImageFrequency`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/image_frequency.md) value describing how often image data will be captured.

        :return: A list of commands that must be sent to end any action.
        """

        if self._sub_action is None:
            return super().get_end_commands(resp=resp, static=static, dynamic=dynamic, image_frequency=image_frequency)
        else:
            # Stop the last sub-action.
            return self._sub_action.get_end_commands(resp=resp, static=static, dynamic=dynamic, image_frequency=image_frequency)