# PutIn

`from transport_challenge_multi_agent.put_in import PutIn`

Put an object in a container.

The Replicant must already be holding the container in one hand and the object in the other hand.

Internally, this action calls several "sub-actions", each a Replicant `Action`:

1. `ReachFor` to move the hand holding the object away from center.
2. `ReachFor` to move the hand holding the container to be in front of the Replicant.
3. `ReachFor` to move the hand holding the object to be above the container.
4. `Drop` to drop the object into the container.
5. [`ResetArms`](reset_arms.md) to reset both arms.

If the object lands in the container, the action ends in success and the object is then made kinematic and parented to the container.

***

## Functions

#### \_\_init\_\_

**`PutIn(dynamic, state)`**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| dynamic |  ReplicantDynamic |  | The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call. |
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

