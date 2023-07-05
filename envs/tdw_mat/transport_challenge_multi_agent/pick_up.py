from enum import Enum
from typing import List
import numpy as np
from tdw.replicant.arm import Arm
from tdw.replicant.action_status import ActionStatus
from tdw.replicant.replicant_static import ReplicantStatic
from tdw.replicant.replicant_dynamic import ReplicantDynamic
from tdw.replicant.image_frequency import ImageFrequency
from tdw.replicant.actions.turn_to import TurnTo
from tdw.replicant.actions.grasp import Grasp
from transport_challenge_multi_agent.challenge_state import ChallengeState
from transport_challenge_multi_agent.reach_for_transport_challenge import ReachForTransportChallenge
from transport_challenge_multi_agent.reset_arms import ResetArms
from transport_challenge_multi_agent.multi_action import MultiAction


class _PickUpState(Enum):
    """
    Enum values describing the state of a `PickUp` action.
    """

    turning_to = 0
    reaching_for = 1
    grasping = 2
    resetting = 3


class PickUp(MultiAction):
    """
    A combination of `TurnTo` + `ReachFor` + `Grasp` + [`ResetArms`](reset_arms.md).
    """

    def __init__(self, arm: Arm, target: int, state: ChallengeState):
        """
        :param arm: The [`Arm`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/arm.md) picking up the object.
        :param target: The object ID.
        :param state: The [`ChallengeState`](challenge_state.md) data.
        """

        super().__init__(state=state)
        self._arm: Arm = arm
        self._target: int = target
        self._pick_up_state: _PickUpState = _PickUpState.turning_to
        self._end_status: ActionStatus = ActionStatus.success

    def get_initialization_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic,
                                    image_frequency: ImageFrequency) -> List[dict]:
        """
        :param resp: The response from the build.
        :param static: The [`ReplicantStatic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_static.md) data that doesn't change after the Replicant is initialized.
        :param dynamic: The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call.
        :param image_frequency: An [`ImageFrequency`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/image_frequency.md) value describing how often image data will be captured.

        :return: A list of commands to initialize this action.
        """

        # Is a Replicant already holding this object?
        if not self._can_pick_up(resp=resp, static=static, dynamic=dynamic) or self._arm in dynamic.held_objects:
            self.status = ActionStatus.cannot_grasp
            return super().get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                       image_frequency=image_frequency)
        # Turn to face the object.
        else:
            self._image_frequency = image_frequency
            self._sub_action = TurnTo(target=self._target)
            return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                                image_frequency=image_frequency)

    def get_ongoing_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        """
        Evaluate an action per-frame to determine whether it's done.

        :param resp: The response from the build.
        :param static: The [`ReplicantStatic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_static.md) data that doesn't change after the Replicant is initialized.
        :param dynamic: The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call.

        :return: A list of commands to send to the build to continue the action.
        """

        # Check if another Replicant is holding the object.
        if self._pick_up_state != _PickUpState.resetting and not self._can_pick_up(resp=resp, static=static, dynamic=dynamic):
            self._end_status = ActionStatus.cannot_grasp
            return self._reset(resp=resp, static=static, dynamic=dynamic)
        # Continue an ongoing sub-action.
        if self._sub_action.status == ActionStatus.ongoing:
            return self._sub_action.get_ongoing_commands(resp=resp, static=static, dynamic=dynamic)
        elif self._sub_action.status == ActionStatus.success:
            # Reach for the object.
            if self._pick_up_state == _PickUpState.turning_to:
                return self._reach_for(resp=resp, static=static, dynamic=dynamic)
            # Grasp the object.
            elif self._pick_up_state == _PickUpState.reaching_for:
                return self._grasp(resp=resp, static=static, dynamic=dynamic)
            # Reset the arm.
            elif self._pick_up_state == _PickUpState.grasping:
                return self._reset(resp=resp, static=static, dynamic=dynamic)
            # We're done!
            elif self._pick_up_state == _PickUpState.resetting:
                self.status = self._end_status
                return []
            else:
                raise Exception(self._pick_up_state)
        # We failed.
        else:
            # Remember the fail status.
            self._end_status = self._sub_action.status
            return self._reset(resp=resp, static=static, dynamic=dynamic)

    def _can_pick_up(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> bool:
        """
        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.

        :return: True if we can pick up this object. An object can't be picked up if it is already grasped or contained.
        """

        for replicant_id in self._state.replicants:
            if replicant_id == static.replicant_id:
                continue
            for arm in [Arm.left, Arm.right]:
                if self._target == self._state.replicants[replicant_id][arm]:
                    return False
        for container_id in self._state.containment:
            if self._target in self._state.containment[container_id]:
                return False
        # Is the object too far away?
        object_position = self._get_object_position(object_id=self._target, resp=resp)
        if np.linalg.norm(dynamic.transform.position - object_position) > 1.5:
            return False
        else:
            return True

    def _reach_for(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        """
        Start to reach for the object.

        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.

        :return: A list of commands.
        """

        self._pick_up_state = _PickUpState.reaching_for
        self._sub_action = ReachForTransportChallenge(target=self._target,
                                                      arm=self._arm,
                                                      dynamic=dynamic,
                                                      absolute=True)
        return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                            image_frequency=self._image_frequency)

    def _grasp(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        """
        Start to grasp the object.

        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.

        :return: A list of commands.
        """

        self._sub_action = Grasp(target=self._target,
                                 arm=self._arm,
                                 dynamic=dynamic,
                                 angle=0,
                                 axis="pitch",
                                 offset=0,
                                 relative_to_hand=True)
        self._pick_up_state = _PickUpState.grasping
        return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                            image_frequency=self._image_frequency)

    def _reset(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        """
        Start to reset the arm.

        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.

        :return: A list of commands.
        """

        self._sub_action = ResetArms(state=self._state)
        self._pick_up_state = _PickUpState.resetting
        return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                            image_frequency=self._image_frequency)
