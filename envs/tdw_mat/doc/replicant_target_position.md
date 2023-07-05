# ReplicantTargetPosition

`from transport_challenge_multi_agent.replicant_target_position import ReplicantTargetPosition`

Enum values describing a target position for a Replicant's hand.

| Value | Description |
| --- | --- |
| `pick_up_end_left` | During a `PickUp` left-handed action, reset the left hand to this position. |
| `pick_up_end_right` | During a `PickUp` right-handed action, reset the right hand to this position. |
| `put_in_move_away_left` | During a `PutIn` action, if the left hand is moving the target object, move the left hand to this position. |
| `put_in_move_away_right` | During a `PutIn` action, if the right hand is moving the target object, move the right hand to this position. |
| `put_in_container_in_front` | During a `PutIn` action, move the container to this position. |