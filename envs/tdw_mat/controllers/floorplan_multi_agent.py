from enum import Enum
from typing import List, Tuple
import numpy as np
from tdw.replicant.action_status import ActionStatus
from tdw.replicant.image_frequency import ImageFrequency
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from transport_challenge_multi_agent.transport_challenge import TransportChallenge


class _AgentState(Enum):
    """
    A simple enum state machine for Replicant agents in this example script.
    """

    waiting = 0  # The Replicant can't do an action right now.
    navigating_to_container = 1  # The Replicant is navigating to a container.
    picking_up_container = 2  # The Replicant is picking up a container.
    navigating_to_target_object = 3  # The Replicant is navigating to a target object.
    picking_up_target_object = 4  # The Replicant is picking up a target object.
    putting_target_object_in_container = 5  # The Replicant is putting a target object in a container.
    backing_up = 6  # The Replicant got to close to an object or collided with an object and is now backing up.
    done = 7  # The Replicant is done.


class FloorplanMultiAgent(TransportChallenge):
    """
    An example of how to use simple state machines to drive multiple agents to pick up containers and target objects.

    The trial ends when both Replicants are "done". A Replicant is "done" if:

    1. It has put a target object in a container.
    2. Or, it has failed to move after 100 attempts.
    """

    def __init__(self, port: int = 1071, check_version: bool = True, launch_build: bool = True, screen_width: int = 256,
                 screen_height: int = 256, image_frequency: ImageFrequency = ImageFrequency.once, png: bool = True,
                 image_passes: List[str] = None):
        super().__init__(port=port, check_version=check_version, launch_build=launch_build, screen_width=screen_width,
                         screen_height=screen_height, image_frequency=image_frequency, png=png, image_passes=image_passes)
        self.states: List[_AgentState] = list()
        self.navigation_attempts: List[int] = list()

    def trial(self, scene: str, layout: int, random_seed: int = None) -> None:
        self.start_floorplan_trial(scene=scene, layout=layout, replicants=2, num_containers=4, num_target_objects=8,
                                   random_seed=random_seed)
        # Add a third-person camera.
        camera = ThirdPersonCamera(avatar_id="a",
                                   position={"x": 0, "y": 20, "z": 0},
                                   look_at={"x": 0, "y": 0, "z": 0})
        self.add_ons.append(camera)
        self.communicate({"$type": "set_floorplan_roof",
                          "show": False})
        # Set initial states.
        self.states.clear()
        self.states.extend([_AgentState.waiting for _ in range(len(self.replicants))])
        self.navigation_attempts.clear()
        self.navigation_attempts = [0 for _ in range(len(self.replicants))]
        # Start a loop. The simulation ends when there are no more target objects to pick up.
        done = False
        while not done:
            for i in self.replicants:
                self._evaluate_replicant(i=i)
            done = len([s for s in self.states if s == _AgentState.done]) == len(self.states)
            self.communicate([])
        self.communicate({"$type": "terminate"})

    def _evaluate_replicant(self, i: int) -> None:
        # Start a new action.
        if self.states[i] == _AgentState.waiting:
            self._get_object(i=i)
        # The Replicant failed to start to navigate to a container or target object.
        if self.states[i] == _AgentState.waiting:
            return
        # Continue an ongoing action.
        if self.replicants[i].action.status == ActionStatus.ongoing:
            return
        elif self.states[i] == _AgentState.done:
            return
        # The action succeeded. Do the next action.
        elif self.replicants[i].action.status == ActionStatus.success:
            self.navigation_attempts[i] = 0
            # Done navigating to the container.
            if self.states[i] == _AgentState.navigating_to_container:
                self._pick_up_container(i=i)
            # Done picking up the container.
            elif self.states[i] == _AgentState.picking_up_container:
                self._navigate_to_target_object(i=i)
            # Done navigating to the target object.
            elif self.states[i] == _AgentState.navigating_to_target_object:
                self._pick_up_target_object(i=i)
            # Done picking up the target object.
            elif self.states[i] == _AgentState.picking_up_target_object:
                self._put_in(i=i)
            # Done putting the target object in a container.
            elif self.states[i] == _AgentState.putting_target_object_in_container:
                self.states[i] = _AgentState.done
            # Done backing up. Try moving again.
            elif self.states[i] == _AgentState.backing_up:
                self._get_object(i=i)
            else:
                raise Exception(self.states[i])
        # The action failed due to a collision. Back up.
        elif self.replicants[i].action.status == ActionStatus.collision:
            self._back_up(i=i)
        # The action failed.
        else:
            # Failed to navigate.
            if self.states[i] == _AgentState.navigating_to_target_object or self.states[i] == _AgentState.navigating_to_container or self.states[i] == _AgentState.backing_up:
                self.navigation_attempts[i] += 1
            # Stop trying.
            if self.navigation_attempts[i] >= 100:
                self.states[i] = _AgentState.done
            # Try again.
            else:
                self.states[i] = _AgentState.waiting
        return

    def _get_object(self, i: int) -> None:
        # The Replicant isn't holding a container. Navigate to a container.
        if not self.state.is_holding_container(replicant_id=self.replicants[i].replicant_id):
            self._navigate_to_container(i=i)
        # The Replicant isn't holding a target object. Navigate to a target object.
        elif not self.state.is_holding_target_object(replicant_id=self.replicants[i].replicant_id):
            self._navigate_to_target_object(i=i)

    def _navigate_to_container(self, i: int) -> None:
        # Try to find a nearby available container.
        got_container, container_id = self._get_nearest_object(object_ids=self.state.container_ids, i=i)
        # Start to navigate to the container.
        if got_container:
            self.states[i] = _AgentState.navigating_to_container
            self.replicants[i].navigate_to(self.object_manager.transforms[container_id].position)
        else:
            self.states[i] = _AgentState.waiting

    def _navigate_to_target_object(self, i: int) -> None:
        # Try to find a nearby available target object.
        got_target_object, target_object_id = self._get_nearest_object(object_ids=self.state.target_object_ids, i=i)
        # Start to navigate to the target object.
        if got_target_object:
            self.states[i] = _AgentState.navigating_to_target_object
            self.replicants[i].navigate_to(self.object_manager.transforms[target_object_id].position)
        else:
            self.states[i] = _AgentState.waiting

    def _pick_up_container(self, i: int) -> None:
        got_container, container_id = self._get_nearest_object(object_ids=self.state.container_ids, i=i)
        # Pick up the container.
        if got_container:
            self.replicants[i].pick_up(target=container_id)
            self.states[i] = _AgentState.picking_up_container
        # Wait and try again.
        else:
            self.states[i] = _AgentState.waiting

    def _pick_up_target_object(self, i: int) -> None:
        if self.state.is_holding_target_object(self.replicants[i].replicant_id):
            self._put_in(i=i)
            return
        got_target_object, target_object_id = self._get_nearest_object(object_ids=self.state.target_object_ids, i=i)
        # Pick up the container.
        if got_target_object:
            self.replicants[i].pick_up(target=target_object_id)
            self.states[i] = _AgentState.picking_up_target_object
        # Wait and try again.
        else:
            self.states[i] = _AgentState.waiting

    def _put_in(self, i: int) -> None:
        self.replicants[i].put_in()
        self.states[i] = _AgentState.putting_target_object_in_container

    def _back_up(self, i: int) -> None:
        # Get the centroid of the positions of the objects we collided with.
        collision_ids = []
        for body_part_id in self.replicants[i].dynamic.collisions:
            collision_ids.extend(self.replicants[i].dynamic.collisions[body_part_id])
        collision_ids = list(set(collision_ids))
        collision_positions = []
        for object_id in collision_ids:
            if object_id in self.object_manager.transforms:
                collision_positions.append(self.object_manager.transforms[object_id].position)
            elif object_id in self.replicants:
                collision_positions.append(self.replicants[object_id].dynamic.transform.position)
        centroid = np.mean(np.array(collision_positions).reshape(-1, 3), axis=0)
        centroid[1] = 0
        # Move away from the centroid.
        position = np.copy(self.replicants[i].dynamic.transform.position)
        position[1] = 0
        v = (position - centroid)
        v = v / np.linalg.norm(v)
        position += v * 0.5
        self.replicants[i].move_to(target=position)
        self.states[i] = _AgentState.backing_up

    def _get_nearest_object(self, object_ids: List[int], i: int) -> Tuple[bool, int]:
        # Ignore contained objects.
        contained = []
        for container_id in self.state.containment:
            contained.extend(self.state.containment[container_id])
        o_ids = [o_id for o_id in object_ids if o_id not in contained]
        # Get the nearest object.
        nearest_object_id = -1
        nearest_distance = 1000
        got_object = False
        replicant_position: np.ndarray = self.replicants[i].dynamic.transform.position
        for object_id in o_ids:
            held = False
            for replicant_id in self.state.replicants:
                for arm in self.state.replicants[replicant_id]:
                    if self.state.replicants[replicant_id][arm] == object_id:
                        held = True
                        break
                if held:
                    break
            # Ignore a held object.
            if held:
                continue
            got_object = True
            # Get the position of the object.
            object_position = self.object_manager.transforms[object_id].position
            # Get the distance from the Replicant to the object.
            distance = np.linalg.norm(replicant_position - object_position)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_object_id = object_id
        return got_object, nearest_object_id


if __name__ == "__main__":
    c = FloorplanMultiAgent(screen_width=1280, screen_height=720)
    c.trial(scene="2a", layout=0, random_seed=1)
