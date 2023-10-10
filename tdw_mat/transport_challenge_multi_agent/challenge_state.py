from typing import Dict, List, Optional
from tdw.replicant.arm import Arm
from tdw.output_data import OutputData, Replicants, Containment
from tdw.add_ons.add_on import AddOn


class ChallengeState(AddOn):
    """
    An add-on that tracks scene-state data that all Replicants need to reference to complete the challenge.
    """

    def __init__(self):
        """
        (no parameters)
        """

        super().__init__()
        """:field
        A dictionary of Replicant data. Key = The ID of the Replicant. Value = A dictionary. Key = [`Arm`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/arm.md). Value = Object ID or None. Example: Replicant 0 is holding object 1 in its right hand. This dictionary will be: `{0: {Arm.left: None, Arm.right: 1}}` 
        """
        self.replicants: Dict[int, Dict[Arm, Optional[int]]] = dict()
        """:field
        A list of the ID of each container in the scene.
        """
        self.container_ids: List[int] = list()
        """:field
        A list of the ID of each target object in the scene.
        """
        self.target_object_ids: List[int] = list()
        """:field
        A dictionary describing the current containment status of each container in the scene. Key = The object ID of a container. Value = A list of IDs of objects inside the container.
        """
        self.containment: Dict[int, List[int]] = dict()
        self.__initialized: bool = False

    def is_holding_container(self, replicant_id: int) -> bool:
        """
        A helper function for determining if a Replicant is holding a container.

        :param replicant_id: The Replicant ID.

        :return: True if the Replicant is holding a container in either hand.
        """

        for arm in self.replicants[replicant_id]:
            object_id = self.replicants[replicant_id][arm]
            if object_id in self.container_ids:
                return True
        return False

    def is_holding_target_object(self, replicant_id: int) -> bool:
        """
        A helper function for determining if a Replicant is holding a target object.

        :param replicant_id: The Replicant ID.

        :return: True if the Replicant is holding a target object in either hand.
        """

        for arm in self.replicants[replicant_id]:
            object_id = self.replicants[replicant_id][arm]
            if object_id in self.target_object_ids:
                return True
        return False

    def get_initialization_commands(self) -> List[dict]:
        """
        This function gets called exactly once per add-on. To re-initialize, set `self.initialized = False`.

        :return: A list of commands that will initialize this add-on.
        """

        return [{"$type": "send_containment",
                 "frequency": "always"},
                {"$type": "send_replicants",
                 "frequency": "always"}]

    def on_send(self, resp: List[bytes]) -> None:
        """
        This is called within `Controller.communicate(commands)` after commands are sent to the build and a response is received.

        Use this function to send commands to the build on the next `Controller.communicate(commands)` call, given the `resp` response.
        Any commands in the `self.commands` list will be sent on the *next* `Controller.communicate(commands)` call.

        :param resp: The response from the build.
        """

        if not self.__initialized:
            self.__initialized = True
            for i in range(len(resp) - 1):
                r_id = OutputData.get_data_type_id(resp[i])
                # Get the Replicants.
                if r_id == "repl":
                    replicants = Replicants(resp[i])
                    for j in range(replicants.get_num()):
                        self.replicants[replicants.get_id(j)] = {Arm.left: None, Arm.right: None}
        # Clear containment.
        self.containment.clear()
        # Read per-frame data.
        for i in range(len(resp) - 1):
            r_id = OutputData.get_data_type_id(resp[i])
            # Get the Replicants.
            if r_id == "repl":
                replicants = Replicants(resp[i])
                for j in range(replicants.get_num()):
                    replicant_id = replicants.get_id(j)
                    # Get the held objects.
                    if replicants.get_is_holding_left(j):
                        self.replicants[replicant_id][Arm.left] = replicants.get_held_left(j)
                    else:
                        self.replicants[replicant_id][Arm.left] = None
                    if replicants.get_is_holding_right(j):
                        self.replicants[replicant_id][Arm.right] = replicants.get_held_right(j)
                    else:
                        self.replicants[replicant_id][Arm.right] = None
            # Get containment.
            elif r_id == "cont":
                containment = Containment(resp[i])
                self.containment[containment.get_object_id()] = [int(o) for o in containment.get_overlap_ids() if o not in self.replicants]

    def reset(self) -> None:
        """
        Reset the `ChallengeState`.
        """

        self.__initialized = False
        self.initialized = False
        self.replicants.clear()
        self.containment.clear()
        self.container_ids.clear()
        self.target_object_ids.clear()
