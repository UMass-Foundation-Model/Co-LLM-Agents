from enum import Enum
from typing import Dict


class ReplicantTargetPosition(Enum):
    """
    Enum values describing a target position for a Replicant's hand.
    """

    pick_up_end_left = 0  # During a `PickUp` left-handed action, reset the left hand to this position.
    pick_up_end_right = 1  # During a `PickUp` right-handed action, reset the right hand to this position.
    put_in_move_away_left = 2  # During a `PutIn` action, if the left hand is moving the target object, move the left hand to this position.
    put_in_move_away_right = 3  # During a `PutIn` action, if the right hand is moving the target object, move the right hand to this position.
    put_in_container_in_front = 4  # During a `PutIn` action, move the container to this position.


def __get_positions() -> Dict[ReplicantTargetPosition, Dict[str, float]]:
    """
    :return: A dictionary. Key = `ReplicantTargetPosition`. Value = The relative position.
    """

    positions: Dict[ReplicantTargetPosition, Dict[str, float]] = dict()
    for x, arm in zip([-0.2, 0.2], [ReplicantTargetPosition.pick_up_end_left, ReplicantTargetPosition.pick_up_end_right]):
        positions[arm] = {"x": x, "y": 1, "z": 0.55}
    for x, arm in zip([-0.45, 0.45], [ReplicantTargetPosition.put_in_move_away_left, ReplicantTargetPosition.put_in_move_away_right]):
        positions[arm] = {"x": x, "y": 1, "z": 0.55}
    positions[ReplicantTargetPosition.put_in_container_in_front] = {"x": 0, "y": 1.15, "z": 0.55}
    return positions

# The relative positions. Key = `ReplicantTargetPosition`. Value = The relative position.
POSITIONS: Dict[ReplicantTargetPosition, Dict[str, float]] = __get_positions()