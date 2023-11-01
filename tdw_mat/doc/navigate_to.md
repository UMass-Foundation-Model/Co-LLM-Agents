# NavigateTo

`from transport_challenge_multi_agent.navigate_to import NavigateTo`

Navigate to a target object or position.

This action requests a NavMesh path. Then, it "snaps" each point on the path to a position with more free space around it.

Then, this action calls `MoveTo` sub-actions to each point on the path.

This action can fail if:

1. The pathfinding fails (for example, if the target position is inside furniture).
2. Any of the `MoveTo` actions fail (for example, if there is a collision).

Additionally, the "point-snapping" may occasionally choose a bad position, especially in densely-populated environments.

***

## Functions

#### \_\_init\_\_

**`NavigateTo(target, collision_detection, state)`**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| target |  Union[int, Dict[str, float] |  | The target. If int: An object ID. If dict: A position as an x, y, z dictionary. If numpy array: A position as an [x, y, z] numpy array. |
| collision_detection |  CollisionDetection |  | The [`CollisionDetection`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/collision_detection.md) rules. |
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

