# ResetArms

`from transport_challenge_multi_agent.reset_arms import ResetArms`

Reset both arms, one after the other.

If an arm is holding an object, it resets with to a position holding the object in front of the Replicant.

If the arm isn't holding an object, it resets to its neutral position.

***

## Functions

#### \_\_init\_\_

**`ResetArms(state)`**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| state |  ChallengeState |  | The [`ChallengeState`](challenge_state.md) data. |

#### get_initialization_commands

**`self.get_initialization_commands(resp, static, dynamic, image_frequency)`**


| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| resp |  List[bytes] |  | The response from the build. |
| static |  ReplicantStatic |  | The [`ReplicantStatic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_static.md) data that doesn't change after the Replicant is initialized. |
| dynamic |  ReplicantDynamic |  | The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call. |
| image_frequency |  ImageFrequency |  | An [`ImageFrequency`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/image_frequency.md) value describing how often image data will be captured. |

_Returns:_  A list of commands to initialize this action.

#### get_ongoing_commands

**`self.get_ongoing_commands(resp, static, dynamic)`**

Evaluate an action per-frame to determine whether it's done.


| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| resp |  List[bytes] |  | The response from the build. |
| static |  ReplicantStatic |  | The [`ReplicantStatic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_static.md) data that doesn't change after the Replicant is initialized. |
| dynamic |  ReplicantDynamic |  | The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call. |

_Returns:_  A list of commands to send to the build to continue the action.

