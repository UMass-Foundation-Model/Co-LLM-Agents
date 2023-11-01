# ChallengeState

`from transport_challenge_multi_agent.challenge_state import ChallengeState`

An add-on that tracks scene-state data that all Replicants need to reference to complete the challenge.

***

## Fields

- `replicants` A dictionary of Replicant data. Key = The ID of the Replicant. Value = A dictionary. Key = [`Arm`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/arm.md). Value = Object ID or None. Example: Replicant 0 is holding object 1 in its right hand. This dictionary will be: `{0: {Arm.left: None, Arm.right: 1}}`

- `container_ids` A list of the ID of each container in the scene.

- `target_object_ids` A list of the ID of each target object in the scene.

- `containment` A dictionary describing the current containment status of each container in the scene. Key = The object ID of a container. Value = A list of IDs of objects inside the container.

- `replicant_target_positions` A dictionary of target positions per Replicant. This is used internally for certain actions. Key = Replicant ID. Value = A dictionary. Key = [`ReplicantTargetPosition`](replicant_target_position.md). Value = A position as a numpy array.

***

## Functions

#### \_\_init\_\_

**`ChallengeState()`**

(no parameters)

#### is_holding_container

**`self.is_holding_container(replicant_id)`**

A helper function for determining if a Replicant is holding a container.


| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| replicant_id |  int |  | The Replicant ID. |

_Returns:_  True if the Replicant is holding a container in either hand.

#### is_holding_target_object

**`self.is_holding_target_object(replicant_id)`**

A helper function for determining if a Replicant is holding a target object.


| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| replicant_id |  int |  | The Replicant ID. |

_Returns:_  True if the Replicant is holding a target object in either hand.

#### get_initialization_commands

**`self.get_initialization_commands()`**

This function gets called exactly once per add-on. To re-initialize, set `self.initialized = False`.

_Returns:_  A list of commands that will initialize this add-on.

#### on_send

**`self.on_send(resp)`**

This is called within `Controller.communicate(commands)` after commands are sent to the build and a response is received.

Use this function to send commands to the build on the next `Controller.communicate(commands)` call, given the `resp` response.
Any commands in the `self.commands` list will be sent on the *next* `Controller.communicate(commands)` call.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| resp |  List[bytes] |  | The response from the build. |

#### reset

**`self.reset()`**

Reset the `ChallengeState`.

