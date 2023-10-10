from typing import List, Optional
from tdw.replicant.arm import Arm
from tdw.replicant.action_status import ActionStatus
from tdw.replicant.replicant_static import ReplicantStatic
from tdw.replicant.replicant_dynamic import ReplicantDynamic
from tdw.replicant.image_frequency import ImageFrequency
from tdw.replicant.collision_detection import CollisionDetection
from tdw.replicant.actions.action import Action
from tdw.replicant.actions.reset_arm import ResetArm
from transport_challenge_multi_agent.challenge_state import ChallengeState
from transport_challenge_multi_agent.reach_for_transport_challenge import ReachForTransportChallenge
from transport_challenge_multi_agent.replicant_target_position import ReplicantTargetPosition, POSITIONS
from transport_challenge_multi_agent.globals import Globals


class ResetArms(Action):
    """
    Reset both arms, one after the other.

    If an arm is holding an object, it resets with to a position holding the object in front of the Replicant.

    If the arm isn't holding an object, it resets to its neutral position.
    """

    def __init__(self, state: ChallengeState):
        """
        :param state: The [`ChallengeState`](challenge_state.md) data.
        """

        super().__init__()
        self._state: ChallengeState = state
        self._arm: Arm = Arm.left
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
        return self._reset_arm(resp=resp, dynamic=dynamic, static=static)

    def get_ongoing_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        """
        Evaluate an action per-frame to determine whether it's done.

        :param resp: The response from the build.
        :param static: The [`ReplicantStatic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_static.md) data that doesn't change after the Replicant is initialized.
        :param dynamic: The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call.

        :return: A list of commands to send to the build to continue the action.
        """

        # Continue the sub-action.
        if self._sub_action.status == ActionStatus.ongoing:
            return self._sub_action.get_ongoing_commands(resp=resp, static=static, dynamic=dynamic)
        # One of the arms finished resetting.
        elif self._sub_action.status == ActionStatus.success:
            # We finished resetting the left arm. Now, reset the right arm.
            if self._arm == Arm.left:
                self._arm = Arm.right
                return self._reset_arm(resp=resp, dynamic=dynamic, static=static)
            # We finished resetting the right arm. We're done.
            else:
                self.status = ActionStatus.success
                return self._sub_action.get_ongoing_commands(resp=resp, static=static, dynamic=dynamic)
        # One of the sub-actions failed, so this action failed too.
        else:
            self.status = self._sub_action.status
            return self._sub_action.get_ongoing_commands(resp=resp, static=static, dynamic=dynamic)

    def _reset_arm(self, resp: List[bytes], dynamic: ReplicantDynamic, static: ReplicantStatic) -> List[dict]:
        """
        Start to reset an arm. Set the sub-action.

        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.

        :return: A list of commands.
        """

        # This arm is holding an object. Reset the arm to its hold-neutral position.
        if self._arm in dynamic.held_objects:
            # Get the position of the empty object target position.
            target_position = ReplicantTargetPosition.pick_up_end_left if self._arm == Arm.left else ReplicantTargetPosition.pick_up_end_right
            # Reach for it.
            self._sub_action = ReachForTransportChallenge(target={k: v for k, v in POSITIONS[target_position].items()},
                                                          arm=self._arm,
                                                          dynamic=dynamic,
                                                          absolute=False)
        # This arm isn't holding an object. Reset the arm to its neutral position.
        else:
            self._sub_action = ResetArm(arms=[self._arm],
                                        dynamic=dynamic,
                                        collision_detection=CollisionDetection(objects=False),
                                        duration=0.25,
                                        scale_duration=Globals.SCALE_IK_DURATION,
                                        previous=None)
        return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                            image_frequency=self._image_frequency)
