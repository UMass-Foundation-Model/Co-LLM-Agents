# ReplicantTransportChallenge

`from transport_challenge_multi_agent.replicant_transport_challenge import ReplicantTransportChallenge`

A wrapper class for `Replicant` for the Transport Challenge.

This class is a subclass of `Replicant`. It includes the entire `Replicant` API plus specialized Transport Challenge actions. Only the Transport Challenge actions are documented here. For the full Replicant documentation, [read this.](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/replicants/overview.md)

![](images/action_space.jpg)

***

## Functions

#### \_\_init\_\_

**`ReplicantTransportChallenge(replicant_id, state, position, rotation)`**

**`ReplicantTransportChallenge(replicant_id, state, position, rotation, image_frequency=ImageFrequency.once)`**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| replicant_id |  int |  | The ID of the Replicant. |
| state |  ChallengeState |  | The [`ChallengeState`](challenge_state.md) data. |
| position |  Union[Dict[str, float] |  | The position of the Replicant as an x, y, z dictionary or numpy array. If None, defaults to `{"x": 0, "y": 0, "z": 0}`. |
| rotation |  Union[Dict[str, float] |  | The rotation of the Replicant in Euler angles (degrees) as an x, y, z dictionary or numpy array. If None, defaults to `{"x": 0, "y": 0, "z": 0}`. |
| image_frequency |  ImageFrequency  | ImageFrequency.once | An [`ImageFrequency`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/image_frequency.md) value that sets how often images are captured. |

#### get_initialization_commands

**`self.get_initialization_commands()`**

This function gets called exactly once per add-on. To re-initialize, set `self.initialized = False`.

_Returns:_  A list of commands that will initialize this add-on.

#### turn_by

**`self.turn_by(angle)`**

Turn the Replicant by an angle.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| angle |  float |  | The target angle in degrees. Positive value = clockwise turn. |

#### turn_to

**`self.turn_to(target)`**

Turn the Replicant to face a target object or position.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| target |  Union[int, Dict[str, float] |  | The target. If int: An object ID. If dict: A position as an x, y, z dictionary. If numpy array: A position as an [x, y, z] numpy array. |

#### drop

**`self.drop(arm)`**

**`self.drop(arm, max_num_frames=100)`**

Drop a held target object.

The action ends when the object stops moving or the number of consecutive `communicate()` calls since dropping the object exceeds `self.max_num_frames`.

When an object is dropped, it is made non-kinematic. Any objects contained by the object are parented to it and also made non-kinematic. For more information regarding containment in TDW, [read this](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/semantic_states/containment.md).

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| arm |  Arm |  | The [`Arm`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/arm.md) holding the object. |
| max_num_frames |  int  | 100 | Wait this number of `communicate()` calls maximum for the object to stop moving before ending the action. |

#### move_forward

**`self.move_forward()`**

**`self.move_forward(distance=0.5)`**

Walk a given distance forward. This calls `self.move_by(distance)`.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| distance |  float  | 0.5 | The distance. |

#### move_backward

**`self.move_backward()`**

**`self.move_backward(distance=0.5)`**

Walk a given distance backward. This calls `self.move_by(-distance)`.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| distance |  float  | 0.5 | The distance. |

#### move_to_object

**`self.move_to_object(target)`**

Move to an object. This calls `self.move_to(target)`.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| target |  int |  | The object ID. |

#### pick_up

**`self.pick_up(target)`**

Reach for an object, grasp it, and bring the arm + the held object to a neutral holding position in from the Replicant.

The Replicant will opt for picking up the object with its right hand. If its right hand is already holding an object, it will try to pick up the object with its left hand.

See: [`PickUp`](pick_up.md)

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| target |  int |  | The object ID. |

#### put_in

**`self.put_in()`**

Put an object in a container.

The Replicant must already be holding the container in one hand and the object in the other hand.

See: [`PutIn`](put_in.md)

#### reset_arms

**`self.reset_arms()`**

Reset both arms, one after the other.

If an arm is holding an object, it resets with to a position holding the object in front of the Replicant.

If the arm isn't holding an object, it resets to its neutral position.

#### navigate_to

**`self.navigate_to(target)`**

Navigate along a path to a target.

See: [`NavigateTo`](navigate_to.md)

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| target |  Union[int, Dict[str, float] |  | The target object or position. |

#### look_up

**`self.look_up()`**

**`self.look_up(angle=15)`**

Look upward by an angle.

The head will continuously move over multiple `communicate()` calls until it is looking at the target.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| angle |  float  | 15 | The angle in degrees. |

#### look_down

**`self.look_down()`**

**`self.look_down(angle=15)`**

Look downward by an angle.

The head will continuously move over multiple `communicate()` calls until it is looking at the target.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| angle |  float  | 15 | The angle in degrees. |

#### reset_head

**`self.reset_head()`**

**`self.reset_head(duration=0.1, scale_duration=True)`**

Reset the head to its neutral rotation.

The head will continuously move over multiple `communicate()` calls until it is at its neutral rotation.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| duration |  float  | 0.1 | The duration of the motion in seconds. |
| scale_duration |  bool  | True | If True, `duration` will be multiplied by `framerate / 60)`, ensuring smoother motions at faster-than-life simulation speeds. |

