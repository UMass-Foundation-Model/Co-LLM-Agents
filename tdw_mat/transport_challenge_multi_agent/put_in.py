from enum import Enum
from typing import List, Optional
import numpy as np
from tdw.replicant.action_status import ActionStatus
from tdw.replicant.arm import Arm
from tdw.replicant.replicant_dynamic import ReplicantDynamic
from tdw.replicant.replicant_static import ReplicantStatic
from tdw.replicant.image_frequency import ImageFrequency
from tdw.replicant.actions.drop import Drop
from tdw.tdw_utils import TDWUtils
from transport_challenge_multi_agent.reach_for_transport_challenge import ReachForTransportChallenge
from transport_challenge_multi_agent.challenge_state import ChallengeState
from transport_challenge_multi_agent.replicant_target_position import ReplicantTargetPosition, POSITIONS
from transport_challenge_multi_agent.reset_arms import ResetArms
from transport_challenge_multi_agent.multi_action import MultiAction


class _PutInState(Enum):
    """
    Enum state values for the `PutIn` action.
    """

    moving_object_arm_elbow_away = 1
    moving_container_in_front_of_replicant = 2
    moving_object_over_container = 3
    dropping_object = 4
    parenting_object_to_container = 5
    resetting = 6


class PutIn(MultiAction):
    """
    Put an object in a container.

    The Replicant must already be holding the container in one hand and the object in the other hand.

    Internally, this action calls several "sub-actions", each a Replicant `Action`:

    1. `ReachFor` to move the hand holding the object away from center.
    2. `ReachFor` to move the hand holding the container to be in front of the Replicant.
    3. `ReachFor` to move the hand holding the object to be above the container.
    4. `Drop` to drop the object into the container.
    5. [`ResetArms`](reset_arms.md) to reset both arms.

    If the object lands in the container, the action ends in success and the object is then made kinematic and parented to the container.
    """

    def __init__(self, dynamic: ReplicantDynamic, state: ChallengeState):
        """
        :param dynamic: The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call.
        :param state: The [`ChallengeState`](challenge_state.md) data.
        """

        super().__init__(state=state)
        # Track the state of the action.
        self._put_in_state: _PutInState = _PutInState.moving_object_arm_elbow_away
        # Make sure we are holding both objects, and that one object is a container and the other is a target object.
        self._container_arm: Optional[Arm] = None
        print(dynamic.held_objects)
        print(state.container_ids)
        print(state.target_object_ids)
        for arm in [Arm.left, Arm.right]:
            if arm in dynamic.held_objects and dynamic.held_objects[arm] in state.container_ids:
                self._container_arm: Arm = arm
                break
        self._object_arm: Optional[Arm] = None
        for arm in [Arm.left, Arm.right]:
            if arm in dynamic.held_objects and dynamic.held_objects[arm] in state.target_object_ids:
                self._object_arm: Arm = arm
                break
        print(self._container_arm, self._object_arm)
        if self._container_arm is None or self._object_arm is None:
            self.status = ActionStatus.not_holding
            self._container_id: int = -1
            self._object_id: int = -1
        else:
            self._container_id = int(dynamic.held_objects[self._container_arm])
            self._object_id = int(dynamic.held_objects[self._object_arm])

    def get_initialization_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic,
                                    image_frequency: ImageFrequency) -> List[dict]:
        """
        :param resp: The response from the build.
        :param static: The [`ReplicantStatic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_static.md) data that doesn't change after the Replicant is initialized.
        :param dynamic: The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call.
        :param image_frequency: An [`ImageFrequency`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/image_frequency.md) value describing how often image data will be captured.

        :return: A list of commands to initialize this action.
        """

        # Remember the image frequency.
        self._image_frequency = image_frequency
        # Move the other arm out of the way.
        return self._move_object_arm_away(resp=resp, static=static, dynamic=dynamic)

    def get_ongoing_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        """
        Evaluate an action per-frame to determine whether it's done.

        :param resp: The response from the build.
        :param static: The [`ReplicantStatic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_static.md) data that doesn't change after the Replicant is initialized.
        :param dynamic: The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call.

        :return: A list of commands to send to the build to continue the action.
        """

        # Continue an ongoing sub-action.
        if self._sub_action.status == ActionStatus.ongoing:
        
            return self._sub_action.get_ongoing_commands(resp=resp, static=static, dynamic=dynamic)
        elif self._sub_action.status == ActionStatus.success or ActionStatus.still_dropping:
            # Move the container in front of the Replicant.
            if self._put_in_state == _PutInState.moving_object_arm_elbow_away:
                print("moving container in front of replicant")
                return self._move_container_in_front_of_replicant(resp=resp, dynamic=dynamic, static=static)
            # Move the object over the container.
            elif self._put_in_state == _PutInState.moving_container_in_front_of_replicant:
                print("moving object over container")
                return self._move_object_over_container(resp=resp, dynamic=dynamic, static=static)
            # Drop the object.
            elif self._put_in_state == _PutInState.moving_object_over_container:
                print("dropping object")
                return self._drop(resp=resp, dynamic=dynamic, static=static)
            # Parent the object to the container.
            elif self._put_in_state == _PutInState.dropping_object:
                print("parenting object to container")
                return self._parent_object(resp=resp)
            # Reset the arms.
            elif self._put_in_state == _PutInState.parenting_object_to_container:
                print("resetting arms")
                return self._reset_arms(resp=resp, dynamic=dynamic, static=static)
            # We're done!
            elif self._put_in_state == _PutInState.resetting:
                self.status = ActionStatus.success
                return []
            else:
                raise Exception(self._put_in_state)
        # The sub-action failed.
        else:
            self.status = self._sub_action.status
            return self._sub_action.get_ongoing_commands(resp=resp, static=static, dynamic=dynamic)

    def _move_object_arm_away(self, resp: List[bytes], dynamic: ReplicantDynamic, static: ReplicantStatic) -> List[dict]:
        """
        Start to move the target object away from the Replicant to allow room for the container arm.

        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.

        :return: A list of commands.
        """

        if self._object_arm == Arm.left:
            target = POSITIONS[ReplicantTargetPosition.put_in_move_away_left]
        else:
            target = POSITIONS[ReplicantTargetPosition.put_in_move_away_right]
        self._sub_action = ReachForTransportChallenge(target={k: v for k, v in target.items()},
                                                      arm=self._object_arm,
                                                      dynamic=dynamic,
                                                      duration=0.05,
                                                      absolute=False)
        return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                            image_frequency=self._image_frequency)

    def _move_container_in_front_of_replicant(self, resp: List[bytes], dynamic: ReplicantDynamic, static: ReplicantStatic) -> List[dict]:
        """
        Start to move the container to be in front of the Replicant.

        Set `self._put_in_state` to `moving_container_in_front_of_replicant`.

        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.

        :return: A list of commands.
        """

        # Get a position in front of the Replicant.
        target = POSITIONS[ReplicantTargetPosition.put_in_container_in_front]
        # Move the position upwards.
        self._sub_action = ReachForTransportChallenge(target={k: v for k, v in target.items()},
                                                      arm=self._container_arm,
                                                      dynamic=dynamic,
                                                      absolute=False)
        self._put_in_state = _PutInState.moving_container_in_front_of_replicant
        return self._sub_action.get_initialization_commands(resp=resp, dynamic=dynamic, static=static,
                                                            image_frequency=self._image_frequency)

    def _move_object_over_container(self, resp: List[bytes], dynamic: ReplicantDynamic, static: ReplicantStatic) -> List[dict]:
        """
        Start to move the target object to be above the container.

        Set `self._put_in_state` to `moving_object_over_container`.

        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.

        :return: A list of commands.
        """

        # Get a position over the container.
        container_position = self._get_object_position(resp=resp, object_id=self._container_id)
        target_position = TDWUtils.array_to_vector3(container_position)
        target_position["y"] += 0.4
        self._sub_action = ReachForTransportChallenge(target=target_position,
                                                      arm=self._object_arm,
                                                      dynamic=dynamic,
                                                      absolute=True)
        self._put_in_state = _PutInState.moving_object_over_container
        return self._sub_action.get_initialization_commands(resp=resp, dynamic=dynamic, static=static,
                                                            image_frequency=self._image_frequency)

    def _drop(self, resp: List[bytes], dynamic: ReplicantDynamic, static: ReplicantStatic) -> List[dict]:
        """
        Start to drop the target object into the container.

        Set `self._put_in_state` to `dropping_object`.

        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.

        :return: A list of commands.
        """

        self._sub_action = Drop(arm=self._object_arm,
                                dynamic=dynamic,
                                max_num_frames=25,
                                offset=0.1)
        self._put_in_state = _PutInState.dropping_object
        return self._sub_action.get_initialization_commands(resp=resp, dynamic=dynamic, static=static,
                                                            image_frequency=self._image_frequency)

    def _parent_object(self, resp: List[bytes]) -> List[dict]:
        """
        Parent the target object to the container.

        If the target object isn't "contained" but it's near the container, teleport it into the container.

        If the container doesn't contain the target object, set `self.status` to `not_holding` and don't parent.

        Otherwise, set `self._put_in_state` to `parenting_object_to_container`.

        :param resp: The response from the build.

        :return: A list of commands.
        """
        self._put_in_state = _PutInState.parenting_object_to_container
        contained = self._container_id in self._state.containment and self._object_id in self._state.containment[self._container_id]
        commands = []
        if not contained:
            container_position = self._get_object_position(object_id=self._container_id, resp=resp)
            target_object_position = self._get_object_position(object_id=self._object_id, resp=resp)
            p0 = np.array([container_position[0], container_position[2]])
            p1 = np.array([target_object_position[0], target_object_position[2]])
            # The object is close enough to snap its position to be in the container.
            print(np.linalg.norm(p0 - p1))
            if np.linalg.norm(p0 - p1) < 0.3:
                teleport_position = TDWUtils.array_to_vector3(container_position)
                # Move the target object slightly above the container.
                teleport_position["y"] += 0.05
                commands.append({"$type": "teleport_object",
                                 "id": self._object_id,
                                 "position": teleport_position})
            # The object is too far away and is not contained.
            else:
                self.status = ActionStatus.not_holding
                return []
        # Parent the target object and make it kinematic.
        commands.extend([{"$type": "parent_object_to_object",
                          "parent_id": self._container_id,
                          "id": self._object_id},
                         {"$type": "set_kinematic_state",
                          "id": self._object_id,
                          "is_kinematic": True,
                          "use_gravity": False}])
        return commands

    def _reset_arms(self, resp: List[bytes], dynamic: ReplicantDynamic, static: ReplicantStatic) -> List[dict]:
        """
        Start to reset the arms.

        Set `self._put_in_state` to `resetting`.

        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.

        :return: A list of commands.
        """

        self._sub_action = ResetArms(state=self._state)
        self._put_in_state = _PutInState.resetting
        return self._sub_action.get_initialization_commands(resp=resp, dynamic=dynamic, static=static,
                                                            image_frequency=self._image_frequency)
