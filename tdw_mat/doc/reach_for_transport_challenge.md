# ReachForTransportChallenge

`from transport_challenge_multi_agent.reach_for_transport_challenge import ReachForTransportChallenge`

A `ReachFor` action with default parameters.

***

## Functions

#### \_\_init\_\_

**`ReachForTransportChallenge(target, dynamic)`**

**`ReachForTransportChallenge(target, dynamic, duration=0.25, offhand_follows=False)`**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| target |  Union[int, Arm, np.ndarray, Dict[str, float] |  | The target. If int: An object ID. If `Arm`: A position in front of one of the sides of the Replicant. If dict: A position as an x, y, z dictionary. If numpy array: A position as an [x, y, z] numpy array. |
| dynamic |  ReplicantDynamic |  | The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call. |
| duration |  float  | 0.25 | The duration of the motion in seconds. |
| offhand_follows |  bool  | False | If True, the offhand will follow the primary hand, meaning that it will maintain the same relative position. Ignored if `len(arms) > 1` or if `target` is an object ID. |

