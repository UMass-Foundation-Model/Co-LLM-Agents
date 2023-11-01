from enum import Enum
from typing import List, Dict, Union, Tuple
import numpy as np
from scipy.spatial import cKDTree
from tdw.replicant.action_status import ActionStatus
from tdw.replicant.collision_detection import CollisionDetection
from tdw.replicant.replicant_static import ReplicantStatic
from tdw.replicant.replicant_dynamic import ReplicantDynamic
from tdw.replicant.image_frequency import ImageFrequency
from tdw.replicant.actions.move_to import MoveTo
from tdw.output_data import OutputData, NavMeshPath, Raycast, IsOnNavMesh
from tdw.tdw_utils import TDWUtils
from transport_challenge_multi_agent.challenge_state import ChallengeState
from transport_challenge_multi_agent.multi_action import MultiAction
from transport_challenge_multi_agent.globals import Globals


class _NavigationState(Enum):
    """
    Enum values describing the state of a `NavigateTo` action.
    """

    getting_nearest_position = 0
    getting_path = 1
    snapping_points = 2
    moving = 3


class NavigateTo(MultiAction):
    """
    Navigate to a target object or position.

    This action requests a NavMesh path. Then, it "snaps" each point on the path to a position with more free space around it.

    Then, this action calls `MoveTo` sub-actions to each point on the path.

    This action can fail if:

    1. The pathfinding fails (for example, if the target position is inside furniture).
    2. Any of the `MoveTo` actions fail (for example, if there is a collision).

    Additionally, the "point-snapping" may occasionally choose a bad position, especially in densely-populated environments.
    """

    _ACCEPTABLE_DISTANCE: float = 0.45
    _ACCEPTABLE_HEIGHT: float = 0.05

    def __init__(self, target: Union[int, Dict[str, float], np.ndarray], collision_detection: CollisionDetection,
                 state: ChallengeState):
        """
        :param target: The target. If int: An object ID. If dict: A position as an x, y, z dictionary. If numpy array: A position as an [x, y, z] numpy array.
        :param collision_detection: The [`CollisionDetection`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/collision_detection.md) rules.
        :param state: The [`ChallengeState`](challenge_state.md) data.
        """

        super().__init__(state=state)
        # The target object or position.
        self.__target:  Union[int, Dict[str, float], np.ndarray] = target
        # The pathfinding target position.
        self._target_position: np.ndarray = np.zeros(shape=3)
        self._navigation_state: _NavigationState = _NavigationState.getting_nearest_position
        self._path: np.ndarray = np.zeros(shape=0)
        self._path_index: int = 1
        self._collision_detection: CollisionDetection = collision_detection

    def get_initialization_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic,
                                    image_frequency: ImageFrequency) -> List[dict]:
        """
        :param resp: The response from the build.
        :param static: The [`ReplicantStatic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_static.md) data that doesn't change after the Replicant is initialized.
        :param dynamic: The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call.
        :param image_frequency: An [`ImageFrequency`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/image_frequency.md) value describing how often image data will be captured.

        :return: A list of commands to initialize this action.
        """

        # Set the target position.
        if isinstance(self.__target, int):
            self._target_position = self._get_object_position(object_id=self.__target, resp=resp)
        elif isinstance(self.__target, np.ndarray):
            self._target_position = np.copy(self.__target)
        elif isinstance(self.__target, dict):
            self._target_position = TDWUtils.vector3_to_array(self.__target)
        else:
            raise Exception(f"Invalid target: {self.__target}")
        self._target_position[1] = 0
        commands = super().get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                       image_frequency=image_frequency)
        commands.append({"$type": "send_is_on_nav_mesh",
                         "position": TDWUtils.array_to_vector3(self._target_position),
                         "id": static.replicant_id})
        return commands

    def get_ongoing_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        """
        Evaluate an action per-frame to determine whether it's done.

        :param resp: The response from the build.
        :param static: The [`ReplicantStatic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_static.md) data that doesn't change after the Replicant is initialized.
        :param dynamic: The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call.

        :return: A list of commands to send to the build to continue the action.
        """

        # Get the NavMesh path.
        if self._navigation_state == _NavigationState.getting_nearest_position:
            return self._get_path(resp=resp, dynamic=dynamic, static=static)
        # Start to snap points.
        elif self._navigation_state == _NavigationState.getting_path:
            for i in range(len(resp) - 1):
                r_id = OutputData.get_data_type_id(resp[i])
                if r_id == "path":
                    path = NavMeshPath(resp[i])
                    if path.get_id() == static.replicant_id:
                        # Failed to pathfind.
                        if path.get_state() != "complete":
                            self.status = ActionStatus.failed_to_move
                            return []
                        # Got a path. Start to snap points.
                        else:
                            self._path = path.get_path()
                            self._navigation_state = _NavigationState.snapping_points
                            return self._spherecast(static=static)
            raise Exception(f"Failed to find a path: {static.replicant_id}")
        # Continue snapping points.
        elif self._navigation_state == _NavigationState.snapping_points:
            return self._snap_point(resp=resp, static=static, dynamic=dynamic)
        # Continue moving.
        elif self._navigation_state == _NavigationState.moving:
            return self._move_to(resp=resp, static=static, dynamic=dynamic)
        else:
            raise Exception(self._navigation_state)

    def _get_path(self, resp: List[bytes], dynamic: ReplicantDynamic, static: ReplicantStatic) -> List[dict]:
        """
        Request NavMeshPath data.

        Set the navigation state to `getting_path`.

        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.
        """

        self._navigation_state = _NavigationState.getting_path
        # Get the nearest point on the NavMesh.
        for i in range(len(resp) - 1):
            r_id = OutputData.get_data_type_id(resp[i])
            if r_id == "isnm":
                is_on_nav_mesh = IsOnNavMesh(resp[i])
                if is_on_nav_mesh.get_is_on() and is_on_nav_mesh.get_id() == static.replicant_id:
                    self._target_position = is_on_nav_mesh.get_position()
                    break
        # Request the path.
        return [{"$type": "send_nav_mesh_path",
                 "origin": TDWUtils.array_to_vector3(dynamic.transform.position),
                 "destination": TDWUtils.array_to_vector3(self._target_position),
                 "id": static.replicant_id}]

    def _snap_point(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        """
        Adjust the current point in the path to be clear of any nearby meshes.

        Then, spherecast at the next point in the path.

        If we're at the last point in the path, start moving.

        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.
        """

        points: List[Tuple[float, float, float]] = list()
        num_hits = 0
        for i in range(len(resp) - 1):
            r_id = OutputData.get_data_type_id(resp[i])
            if r_id == "rayc":
                raycast = Raycast(resp[i])
                # This is my raycast and the raycast hit something. Add the point.
                if raycast.get_raycast_id() == static.replicant_id and raycast.get_hit():
                    num_hits += 1
                    point = raycast.get_point()
                    # Ignore raycasts that hit the floor.
                    if point[1] > NavigateTo._ACCEPTABLE_HEIGHT:
                        points.append(raycast.get_point())
        # There were no raycast hits. Fail now.
        if num_hits == 0:
            self.status = ActionStatus.failed_to_move
            return []
        # There are points. Move away from them.
        if len(points) > 0:
            # Convert the list to a numpy array.
            points_arr: np.ndarray = np.array(points)
            # noinspection PyArgumentList
            nearest_index: int = cKDTree(points_arr).query(self._path[self._path_index], k=1)[1]
            # Get the next point on the path.
            p0 = np.copy(self._path[self._path_index])
            p0[1] = 0
            # Get the nearest point in the cast.
            p1 = np.copy(points_arr[nearest_index])
            p1[1] = 0
            # Get the distance between the two points.
            distance = np.linalg.norm(p0 - p1)
            # There isn't enough space between this point and the nearest obstacle. Adjust the point.
            if 0 < distance < NavigateTo._ACCEPTABLE_DISTANCE:
                dd = NavigateTo._ACCEPTABLE_DISTANCE - distance
                v = (p0 - p1) / distance
                # Adjust the point.
                self._path[self._path_index] += v * dd
        # Increment the path index.
        self._path_index += 1
        # If we've adjusted all points, start moving.
        if self._path_index >= self._path.shape[0] - 1:
            for i in range(self._path.shape[0]):
                self._path[i][1] = 0
            self._navigation_state = _NavigationState.moving
            self._path_index = 0
            return self._move_to(resp=resp, static=static, dynamic=dynamic)
        # Cast a sphere at the next point.
        else:
            return self._spherecast(static=static)

    def _move_to(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        """
        Start to move to the next point in the path.

        If this is the last point, the `NavigateTo` action ends in success.

        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.
        """

        # We haven't started moving or we arrived at the next target.
        if self._sub_action is None or self._sub_action.status == ActionStatus.success:
            # We arrived at the final destination.
            if self._path_index == self._path.shape[0] - 1:
                self.status = ActionStatus.success
                return []
            # Start the next MoveTo sub-action.
            else:
                self._path_index += 1
                # The last point on the path might be an object. If so, we want to approach it but not step on it.
                arrived_at = 0.25 if self._path_index == self._path.shape[0] - 1 and isinstance(self.__target, int) and self.__target in self._state.target_object_ids else 0.1
                # Create the MoveTo action and initialize it.
                self._sub_action = MoveTo(target=self._path[self._path_index],
                                          reset_arms=False,
                                          reset_arms_duration=0.25,
                                          scale_reset_arms_duration=Globals.SCALE_IK_DURATION,
                                          arrived_at=arrived_at,
                                          max_walk_cycles=100,
                                          bounds_position="center",
                                          collision_detection=self._collision_detection,
                                          previous=None)
                return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                                    image_frequency=self._image_frequency)
        else:
            # End in failure.
            if self._sub_action.status != ActionStatus.ongoing:
                self.status = self._sub_action.status
            # Continue.
            return self._sub_action.get_ongoing_commands(resp=resp, static=static, dynamic=dynamic)

    def _spherecast(self, static: ReplicantStatic) -> List[dict]:
        """
        Spherecast at the next point in the path.

        :param static: The static Replicant data.
        """

        origin = TDWUtils.array_to_vector3(self._path[self._path_index])
        origin["y"] = 2.1
        destination = TDWUtils.array_to_vector3(self._path[self._path_index])
        destination["y"] = -2.1
        return [{"$type": "send_spherecast",
                 "radius": NavigateTo._ACCEPTABLE_DISTANCE,
                 "origin": origin,
                 "destination": destination,
                 "id": static.replicant_id}]
