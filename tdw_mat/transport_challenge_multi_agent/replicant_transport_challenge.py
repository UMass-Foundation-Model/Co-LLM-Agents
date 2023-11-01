from typing import Dict, Union
import numpy as np
from tdw.add_ons.replicant import Replicant
from tdw.replicant.image_frequency import ImageFrequency
from tdw.replicant.arm import Arm
from transport_challenge_multi_agent.pick_up import PickUp
from transport_challenge_multi_agent.put_in import PutIn
from transport_challenge_multi_agent.challenge_state import ChallengeState
from transport_challenge_multi_agent.navigate_to import NavigateTo
from transport_challenge_multi_agent.reset_arms import ResetArms
from transport_challenge_multi_agent.globals import Globals


class ReplicantTransportChallenge(Replicant):
    """
    A wrapper class for `Replicant` for the Transport Challenge.

    This class is a subclass of `Replicant`. It includes the entire `Replicant` API plus specialized Transport Challenge actions. Only the Transport Challenge actions are documented here. For the full Replicant documentation, [read this.](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/replicants/overview.md)

    ![](images/action_space.jpg)
    """

    def __init__(self, replicant_id: int, state: ChallengeState,
                 position: Union[Dict[str, float], np.ndarray] = None,
                 rotation: Union[Dict[str, float], np.ndarray] = None,
                 image_frequency: ImageFrequency = ImageFrequency.once,
                 target_framerate: int = 250,
                 enable_collision_detection: bool = False,
                 name: str = "replicant_0",
                 ):
        """
        :param replicant_id: The ID of the Replicant.
        :param state: The [`ChallengeState`](challenge_state.md) data.
        :param position: The position of the Replicant as an x, y, z dictionary or numpy array. If None, defaults to `{"x": 0, "y": 0, "z": 0}`.
        :param rotation: The rotation of the Replicant in Euler angles (degrees) as an x, y, z dictionary or numpy array. If None, defaults to `{"x": 0, "y": 0, "z": 0}`.
        :param image_frequency: An [`ImageFrequency`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/image_frequency.md) value that sets how often images are captured.
        :param target_framerate: The target framerate. It's possible to set a higher target framerate, but doing so can lead to a loss of precision in agent movement.
        """

        super().__init__(replicant_id=replicant_id, position=position, rotation=rotation,
                         image_frequency=image_frequency, target_framerate=target_framerate, name = name)
        self._state: ChallengeState = state
        self.collision_detection.held = False
        self.collision_detection.previous_was_same = False

    def turn_by(self, angle: float) -> None:
        """
        Turn the Replicant by an angle.

        :param angle: The target angle in degrees. Positive value = clockwise turn.
        """

        super().turn_by(angle=angle)

    def turn_to(self, target: Union[int, Dict[str, float], np.ndarray]) -> None:
        """
        Turn the Replicant to face a target object or position.

        :param target: The target. If int: An object ID. If dict: A position as an x, y, z dictionary. If numpy array: A position as an [x, y, z] numpy array.
        """

        super().turn_to(target=target)

    def drop(self, arm: Arm, max_num_frames: int = 100, offset: float = 0.1) -> None:
        """
        Drop a held target object.

        The action ends when the object stops moving or the number of consecutive `communicate()` calls since dropping the object exceeds `self.max_num_frames`.

        When an object is dropped, it is made non-kinematic. Any objects contained by the object are parented to it and also made non-kinematic. For more information regarding containment in TDW, [read this](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/semantic_states/containment.md).

        :param arm: The [`Arm`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/arm.md) holding the object.
        :param max_num_frames: Wait this number of `communicate()` calls maximum for the object to stop moving before ending the action.
        :param offset: Prior to being dropped, set the object's positional offset. This can be a float (a distance along the object's forward directional vector). Or it can be a dictionary or numpy array (a world space position).
        """

        super().drop(arm=arm, max_num_frames=max_num_frames, offset=offset)

    def move_forward(self, distance: float = 0.5) -> None:
        """
        Walk a given distance forward. This calls `self.move_by(distance)`.

        :param distance: The distance.
        """

        super().move_by(distance=abs(distance), reset_arms=False)

    def move_backward(self, distance: float = 0.5) -> None:
        """
        Walk a given distance backward. This calls `self.move_by(-distance)`.

        :param distance: The distance.
        """

        super().move_by(distance=-abs(distance), reset_arms=False)

    def move_to_object(self, target: int) -> None:
        """
        Move to an object. This calls `self.move_to(target)`.

        :param target: The object ID.
        """
        self.move_to(target=target, reset_arms=False, arrived_at=0.7) #if target in self._state.target_object_ids else 0.1)
    
    def move_to_position(self, target: np.ndarray) -> None:
        """
        Move to an position. This calls `self.move_to(target)`.

        :param target: The object ID.
        """
        assert isinstance(target, np.ndarray) and target.shape == (3,), f"target must be a 3D numpy array. Got {target}"
        target[1] = 0
        self.move_to(target=target, reset_arms=False, arrived_at=0.7)
        
    def pick_up(self, target: int) -> None:
        """
        Reach for an object, grasp it, and bring the arm + the held object to a neutral holding position in from the Replicant.

        The Replicant will opt for picking up the object with its right hand. If its right hand is already holding an object, it will try to pick up the object with its left hand.

        See: [`PickUp`](pick_up.md)

        :param target: The object ID.
        """

        self.action = PickUp(arm=Arm.left if Arm.right in self.dynamic.held_objects else Arm.right,
                             target=target, state=self._state)

    def put_in(self) -> None:
        """
        Put an object in a container.

        The Replicant must already be holding the container in one hand and the object in the other hand.

        See: [`PutIn`](put_in.md)
        """

        self.action = PutIn(dynamic=self.dynamic, state=self._state)

    def reset_arms(self) -> None:
        """
        Reset both arms, one after the other.

        If an arm is holding an object, it resets with to a position holding the object in front of the Replicant.

        If the arm isn't holding an object, it resets to its neutral position.
        """

        self.action = ResetArms(state=self._state)

    def navigate_to(self, target: Union[int, Dict[str, float], np.ndarray]) -> None:
        """
        Navigate along a path to a target.

        See: [`NavigateTo`](navigate_to.md)

        :param target: The target object or position.
        """

        self.action = NavigateTo(target=target, collision_detection=self.collision_detection, state=self._state)

    def look_up(self, angle: float = 15) -> None:
        """
        Look upward by an angle.

        The head will continuously move over multiple `communicate()` calls until it is looking at the target.

        :param angle: The angle in degrees.
        """

        self.rotate_head(axis="pitch", angle=-abs(angle), scale_duration=Globals.SCALE_IK_DURATION)

    def look_down(self, angle: float = 15) -> None:
        """
        Look downward by an angle.

        The head will continuously move over multiple `communicate()` calls until it is looking at the target.

        :param angle: The angle in degrees.
        """

        self.rotate_head(axis="pitch", angle=abs(angle), scale_duration=Globals.SCALE_IK_DURATION)

    def reset_head(self, duration: float = 0.1, scale_duration: bool = True) -> None:
        """
        Reset the head to its neutral rotation.

        The head will continuously move over multiple `communicate()` calls until it is at its neutral rotation.

        :param duration: The duration of the motion in seconds.
        :param scale_duration: If True, `duration` will be multiplied by `framerate / 60)`, ensuring smoother motions at faster-than-life simulation speeds.
        """

        super().reset_head(duration=duration, scale_duration=scale_duration)
