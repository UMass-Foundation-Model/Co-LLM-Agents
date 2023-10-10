from csv import DictReader
from typing import List, Dict, Union, Tuple, Optional
from subprocess import run, PIPE
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from tdw.version import __version__
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.floorplan import Floorplan
from tdw.add_ons.replicant import Replicant
from tdw.librarian import HumanoidLibrarian
from tdw.add_ons.occupancy_map import OccupancyMap
from tdw.add_ons.interior_scene_lighting import InteriorSceneLighting
from tdw.add_ons.object_manager import ObjectManager
from tdw.add_ons.nav_mesh import NavMesh
from tdw.replicant.image_frequency import ImageFrequency
from tdw.replicant.action_status import ActionStatus
from tdw.scene_data.scene_bounds import SceneBounds
from transport_challenge_multi_agent.challenge_state import ChallengeState
from transport_challenge_multi_agent.replicant_transport_challenge import ReplicantTransportChallenge
from transport_challenge_multi_agent.paths import CONTAINERS_PATH, TARGET_OBJECTS_PATH, TARGET_OBJECT_MATERIALS_PATH
from transport_challenge_multi_agent.globals import Globals
from transport_challenge_multi_agent.asset_cached_controller import AssetCachedController
from tdw.add_ons.logger import Logger
import os
import json


class TransportChallenge(AssetCachedController):
    """
    A subclass of `Controller` for the Transport Challenge. Always use this class instead of `Controller`.

    See the README for information regarding output data and scene initialization.
    """

    """:class_var
    The mass of each target object.
    """
    TARGET_OBJECT_MASS: float = 0.25
    """:class_var
    If an object is has this mass or greater, the object will become kinematic.
    """
    KINEMATIC_MASS: float = 100
    """:class_var
    The expected version of TDW.
    """
    TDW_VERSION: str = "1.11.18"
    """:class_var
    The goal zone is a circle defined by `self.goal_center` and this radius value.
    """
    GOAL_ZONE_RADIUS: float = 1

    def __init__(self, port: int = 1071, check_version: bool = True, launch_build: bool = True, screen_width: int = 256,
                 screen_height: int = 256, image_frequency: ImageFrequency = ImageFrequency.once, png: bool = True, asset_cache_dir = "transport_challenge_asset_bundles",
                 image_passes: List[str] = None, target_framerate: int = 250, enable_collision_detection: bool = False, new_setting = False, logger_dir = None):
        """
        :param port: The socket port used for communicating with the build.
        :param check_version: If True, the controller will check the version of the build and print the result.
        :param launch_build: If True, automatically launch the build. If one doesn't exist, download and extract the correct version. Set this to False to use your own build, or (if you are a backend developer) to use Unity Editor.
        :param screen_width: The width of the screen in pixels.
        :param screen_height: The height of the screen in pixels.
        :param image_frequency: How often each Replicant will capture an image. `ImageFrequency.once` means once per action, at the end of the action. `ImageFrequency.always` means every communicate() call. `ImageFrequency.never` means never.
        :param png: If True, the image pass from each Replicant will be a lossless png. If False, the image pass from each Replicant will be a lossy jpg.
        :param image_passes: A list of image passes, such as `["_img"]`. If None, defaults to `["_img", "_id", "_depth"]` (i.e. image, segmentation colors, depth maps).
        :param target_framerate: The target framerate. It's possible to set a higher target framerate, but doing so can lead to a loss of precision in agent movement.
        """

        try:
            q = run(["git", "rev-parse", "--show-toplevel"], stdout=PIPE)
            p = Path(str(q.stdout.decode("utf-8").strip())).resolve()
            if p.stem != "Co-LLM-Agents":
                print("Warning! You might be using code copied from the Co-LLM-Agents repo. Your code might be out of date.\n")
        except OSError:
            pass
        if TransportChallenge.TDW_VERSION != __version__:
            print(f"Warning! Your local install of TDW is version {__version__} but the Multi-Agent Transport Challenge requires version {TransportChallenge.TDW_VERSION}\n")
        super().__init__(cache_dir=asset_cache_dir, port=port, check_version=check_version, launch_build=launch_build)
        if logger_dir is not None:
            self.logger = Logger(path=os.path.join(logger_dir, "action_log.log"))
            self.add_ons.append(self.logger)
        else:
            self.logger = None
        self._image_frequency: ImageFrequency = image_frequency
        """:field
        A dictionary of all Replicants in the scene. Key = The Replicant ID. Value = [`ReplicantTransportChallenge`](replicant_transport_challenge.md).
        """
        self.replicants: Dict[int, ReplicantTransportChallenge] = dict()
        """:field
        The `ChallengeState`, which includes container IDs, target object IDs, containment state, and which Replicant is holding which objects.
        """
        self.state: ChallengeState = ChallengeState()
        """:field
        An [`OccupancyMap`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/navigation/occupancy_maps.md). This is used to place objects and can also be useful for navigation.
        """
        self.occupancy_map: OccupancyMap = OccupancyMap()
        """:field
        An [`ObjectManager`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/core_concepts/output_data.md#the-objectmanager-add-on) add-on. This is useful for referencing all objects in the scene.
        """
        self.object_manager: ObjectManager = ObjectManager(transforms=True, rigidbodies=False, bounds=True)
        """:field
         The challenge is successful when the Replicants move all of the target objects to the the goal zone, which is defined by this position and `TransportChallenge.GOAL_ZONE_RADIUS`. This value is set at the start of a trial.
        """
        self.goal_position: np.ndarray = np.zeros(shape=3)
        # Download local asset bundles and set the asset bundle librarian paths.
        # download_asset_bundles()
        # Initialize the random state. This will be reset later.
        self._rng: np.random.RandomState = np.random.RandomState()
        # All possible target objects. Key = name. Value = scale.
        self._target_objects_names_and_scales: Dict[str, float] = dict()
        with open(str(TARGET_OBJECTS_PATH.resolve())) as csvfile:
            reader = DictReader(csvfile)
            for row in reader:
                self._target_objects_names_and_scales[row["name"]] = float(row["scale"])
        self._target_object_names: List[str] = list(self._target_objects_names_and_scales.keys())
        self._target_object_visual_materials: List[str] = TARGET_OBJECT_MATERIALS_PATH.read_text().split("\n")
        # Get all possible container names.
        self._container_names: List[str] = CONTAINERS_PATH.read_text().split("\n")
        self._scene_bounds: Optional[SceneBounds] = None
        if image_passes is None:
            self._image_passes: List[str] = ["_img", "_id", "_depth"]
        else:
            self._image_passes: List[str] = image_passes
        self._target_framerate: int = target_framerate
        self.new_setting = new_setting
        # Initialize the window and rendering.
        self.communicate([{"$type": "set_screen_size",
                           "width": screen_width,
                           "height": screen_height},
                          {"$type": "set_img_pass_encoding",
                           "value": png},
                          {"$type": "set_render_quality",
                           "render_quality": 5}])
        # Sett whether we need to scale IK motion duration.
        Globals.SCALE_IK_DURATION = self._is_standalone
        self.enable_collision_detection = enable_collision_detection
        # Add two replicants: Alice and Bob.
        self.HUMANOID_LIBRARIANS[Replicant.LIBRARY_NAME] = HumanoidLibrarian("transport_challenge_multi_agent/replicants.json")

    def start_floorplan_trial(self, scene: str, layout: int, num_containers: int, num_target_objects: int,
                              container_room_index: int = None, target_objects_room_index: int = None,
                              goal_room_index: int = None, task = None,
                              replicants: Union[int, List[Union[int, np.ndarray, Dict[str, float]]]] = 2,
                              lighting: bool = True, random_seed: int = None, data_prefix = 'dataset/arxiv_dataset') -> None:
        """
        Start a trial in a floorplan scene.

        :param scene: The scene. Options: "1a", "1b", "1c", "2a", "2b", "2c", "4a", "4b", "4c", "5a", "5b", "5c"
        :param layout: The layout of objects in the scene. Options: 0, 1, 2.
        :param num_containers: The number of containers in the scene.
        :param num_target_objects: The number of target objects in the scene.
        :param container_room_index: The index of the room in which containers will be placed. If None, the room is random.
        :param target_objects_room_index: The index of the room in which target objects will be placed. If None, the room is random.
        :param goal_room_index: The index of the goal room. If None, the room is random.
        :param replicants: An integer or a list. If an integer, this is the number of Replicants in the scene; they will be added at random positions on the occupancy map. If a list, each element can be: An integer (the Replicant will be added *in this room* at a random occupancy map position), a numpy array (a worldspace position), or a dictionary (a worldspace position, e.g. `{"x": 0, "y": 0, "z": 0}`.
        :param lighting: If True, add an HDRI skybox for realistic lighting. The skybox will be random, using the `random_seed` value below.
        :param random_seed: The random see used to add containers, target objects, and Replicants, as well as to set the lighting and target object materials. If None, the seed is random.
        """

        floorplan = Floorplan()
        if type(layout) == int:
            floorplan.init_scene(scene=scene, layout=layout) # 0 / 1 / 2
        else:
            floorplan.init_scene(scene=scene, layout=int(layout[0])) # 0_1: the first pos is the layout, the second pos is the container setting
        self.add_ons.append(floorplan)
        if lighting:
            self.add_ons.append(InteriorSceneLighting(rng=np.random.RandomState(random_seed)))
        self.communicate([])
        self.scene = scene
        self.layout = layout
        self.task = task
        self.data_prefix = data_prefix
        print('New Setting:', self.new_setting)
        if self.new_setting:
            self._start_trial_new(replicants=replicants, task_type = task, random_seed=random_seed)
        else:    
            self._start_trial(num_containers=num_containers, num_target_objects=num_target_objects,
                          container_room_index=0, target_objects_room_index=0, goal_room_index=0,
                          replicants=replicants, random_seed=random_seed)

    def start_box_room_trial(self, size: Tuple[int, int], num_containers: int, num_target_objects: int,
                             replicants: Union[int, List[Union[np.ndarray, Dict[str, float]]]] = 2,
                             random_seed: int = None) -> None:
        """
        Start a trial in a simple box room scene.

        :param size: The size of the room.
        :param num_containers: The number of containers in the scene.
        :param num_target_objects: The number of target objects in the scene.
        :param replicants: An integer or a list. If an integer, this is the number of Replicants in the scene; they will be added at random positions on the occupancy map. If a list, each element can be: A numpy array (a worldspace position), or a dictionary (a worldspace position, e.g. `{"x": 0, "y": 0, "z": 0}`.
        :param random_seed: The random see used to add containers, target objects, and Replicants, as well as to set the lighting and target object materials. If None, the seed is random.
        """
        self.communicate([{"$type": "load_scene",
                           "scene_name": "ProcGenScene"},
                          TDWUtils.create_empty_room(size[0], size[1])])
        self._start_trial(num_containers=num_containers, num_target_objects=num_target_objects,
                          container_room_index=0, target_objects_room_index=0, goal_room_index=0,
                          replicants=replicants, random_seed=random_seed)

    def communicate(self, commands: Union[dict, List[dict]]) -> list:
        """
        Send commands and receive output data in response.

        :param commands: A list of JSON commands.

        :return The output data from the build.
        """
        return super().communicate(commands)

    def _start_trial_new(self, replicants: Union[int, List[Union[int, np.ndarray, Dict[str, float]]]] = 2, task_type = 'food', random_seed: int = None) -> None:
        """
        Start a trial in a floorplan scene.
        food or stuff
        """
        self.communicate({"$type": "set_floorplan_roof", "show": False})
        load_path = os.path.join(self.data_prefix, f"{self.scene}_{self.layout}.json")
        with open(load_path, "r") as f: scene = json.load(f)
        if os.path.exists(os.path.join(self.data_prefix, f"{self.scene}_{self.layout}_metadata.json")):
            load_count_and_position_path = os.path.join(self.data_prefix, f"{self.scene}_{self.layout}_metadata.json")
        else:
            load_count_and_position_path = os.path.join(self.data_prefix, f"{self.scene}_{self.layout}_count.json")
        with open(load_count_and_position_path, "r") as f: count_and_position = json.load(f)
        common_sense_path = os.path.join(self.data_prefix, "list.json")
        with open(common_sense_path, "r") as f:
            common_sense = json.load(f)
        self.communicate(scene)
        self.state = ChallengeState()
        self.add_ons.clear() # Clear the add-ons.
        if self.logger is not None:
            self.add_ons.append(self.logger)
        self.replicants.clear()
        # Add an occupancy map.
        self.add_ons.append(self.occupancy_map)
        # Get the rooms.
        rooms: Dict[int, List[Dict[str, float]]] = self._get_rooms_map(communicate=True)
        replicant_positions: List[Dict[str, float]] = list()
        # Spawn a certain number of Replicants in random rooms.
        if isinstance(replicants, int):
            # Randomize the rooms.
            room_indices = list(rooms.keys())
            self._rng.shuffle(room_indices)
            room_index = 0
            # Place Replicants in different rooms.
            for i in range(replicants):
                # Get a random position in the room.
                positions = rooms[room_indices[room_index]]
                replicant_positions.append(positions[self._rng.randint(0, len(positions))])
                room_index += 1
                if room_index >= len(room_indices):
                    room_index = 0
        # Spawn Replicants in the center of rooms or in certain positions.
        elif isinstance(replicants, list):
            for i in range(len(replicants)):
                # Spawn a Replicant at a random position in a room.
                if isinstance(replicants[i], int):
                    positions = rooms[replicants[i]]
                    position = positions[self._rng.randint(0, len(positions))]
                # Spawn a Replicant at a defined position.
                elif isinstance(replicants[i], np.ndarray):
                    position = TDWUtils.array_to_vector3(replicants[i])
                # Spawn a Replicant at a defined position.
                elif isinstance(replicants[i], dict):
                    position = replicants[i]
                else:
                    raise Exception(f"Invalid Replicant position: {replicants[i]}")
                # Add a Replicant position.
                replicant_positions.append(position)
                # Occupy the position.
                rooms = self._occupy_position(position=position)
                
        # Add the Replicants. If the position is fixed, the position is the same as the last time.
        for i in range(len(replicant_positions)):
            if str(i) in count_and_position.keys():
                replicant_positions[i] = count_and_position[str(i)]
            count_and_position[str(i)] = replicant_positions[i]
            
        replicant_names = ["man_suit_edited", "replicant_0"]
        for i, replicant_position in enumerate(replicant_positions):
            replicant_name = replicant_names[i]
            replicant = ReplicantTransportChallenge(replicant_id=i,
                                                    state=self.state,
                                                    position=replicant_position,
                                                    image_frequency=self._image_frequency,
                                                    target_framerate=self._target_framerate,
                                                    enable_collision_detection=self.enable_collision_detection,
                                                    name=replicant_name)
            self.replicants[replicant.replicant_id] = replicant
            self.add_ons.append(replicant)
        # Set the pass masks.
        # Add a challenge state and object manager.
        self.object_manager.reset()
        self.add_ons.extend([self.state, self.object_manager])
        self.communicate([])
        commands = []
        for object_id in self.object_manager.objects_static.keys():
            if self.object_manager.objects_static[object_id].name in common_sense[task_type]['target']:
                self.state.target_object_ids.append(object_id)
            if self.object_manager.objects_static[object_id].name in common_sense[task_type]['container']:
                self.state.container_ids.append(object_id)
        for replicant_id in self.replicants:
            # Set pass masks.
            commands.append({"$type": "set_pass_masks",
                             "pass_masks": self._image_passes,
                             "avatar_id": self.replicants[replicant_id].static.avatar_id})
            # Ignore collisions with target objects.
            self.replicants[replicant_id].collision_detection.exclude_objects.extend(self.state.target_object_ids)
        # Add a NavMesh.
        nav_mesh_exclude_objects = list(self.replicants.keys())
        nav_mesh_exclude_objects.extend(self.state.target_object_ids)
        nav_mesh = NavMesh(exclude_objects=nav_mesh_exclude_objects)
        self.add_ons.append(nav_mesh)
        # Send the commands.
        # self.communicate(commands)
        # Reset the heads.
        for replicant_id in self.replicants:
            self.replicants[replicant_id].reset_head(scale_duration=Globals.SCALE_IK_DURATION)
        reset_heads = False
        while not reset_heads:
            self.communicate([])
            reset_heads = True
            for replicant_id in self.replicants:
                if self.replicants[replicant_id].action.status == ActionStatus.ongoing:
                    reset_heads = False
                    break
        
        '''
        target_object_ids = self.state.target_object_ids
        goal_description = {}
        for i in target_object_ids:
            if object_names[i] in goal_description:
                goal_description[object_names[i]] += 1
            else:
                goal_description[object_names[i]] = 1
        '''

        # Save the position of agents.
        with open(load_count_and_position_path, "w") as f:
            json.dump(count_and_position, f, indent=4)
        print(load_count_and_position_path, 'saved')


    def _start_trial(self, num_containers: int, num_target_objects: int, container_room_index: int = None,
                     target_objects_room_index: int = None, goal_room_index: int = None,
                     replicants: Union[int, List[Union[int, np.ndarray, Dict[str, float]]]] = 2,
                     random_seed: int = None) -> None:
        """
        Start a trial.

        :param num_containers: The number of containers in the scene.
        :param num_target_objects: The number of target objects in the scene.
        :param container_room_index: The index of the room in which containers will be placed. If None, the room is random.
        :param target_objects_room_index: The index of the room in which target objects will be placed. If None, the room is random.
        :param goal_room_index: The index of the goal room. If None, the room is random.
        :param replicants: An integer or a list. If an integer, this is the number of Replicants in the scene; they will be added at random positions on the occupancy map. If a list, each element can be: An integer (the Replicant will be added *in this room* at a random occupancy map position), a numpy array (a worldspace position), or a dictionary (a worldspace position, e.g. `{"x": 0, "y": 0, "z": 0}`.
        :param random_seed: The random see used to add containers, target objects, and Replicants, as well as to set the lighting and target object materials. If None, the seed is random.
        """

        # Create the random state.
        if random_seed is None:
            self._rng = np.random.RandomState()
        else:
            self._rng = np.random.RandomState(random_seed)
        # We haven't created NavMesh obstacles yet.
        Globals.MADE_NAV_MESH_OBSTACLES = False
        # Clear the add-ons.
        self.state = ChallengeState()
        self.add_ons.clear()
        self.replicants.clear()
        # Add an occupancy map.
        self.add_ons.append(self.occupancy_map)
        # Get the rooms.
        rooms: Dict[int, List[Dict[str, float]]] = self._get_rooms_map(communicate=True)
        replicant_positions: List[Dict[str, float]] = list()
        # Spawn a certain number of Replicants in random rooms.
        if isinstance(replicants, int):
            # Randomize the rooms.
            room_indices = list(range(len(rooms)))
            self._rng.shuffle(room_indices)
            room_index = 0
            # Place Replicants in different rooms.
            for i in range(replicants):
                # Get a random position in the room.
                positions = rooms[room_indices[room_index]]
                replicant_positions.append(positions[self._rng.randint(0, len(positions))])
                room_index += 1
                if room_index >= len(room_indices):
                    room_index = 0
        # Spawn Replicants in the center of rooms or in certain positions.
        elif isinstance(replicants, list):
            for i in range(len(replicants)):
                # Spawn a Replicant at a random position in a room.
                if isinstance(replicants[i], int):
                    positions = rooms[replicants[i]]
                    position = positions[self._rng.randint(0, len(positions))]
                # Spawn a Replicant at a defined position.
                elif isinstance(replicants[i], np.ndarray):
                    position = TDWUtils.array_to_vector3(replicants[i])
                # Spawn a Replicant at a defined position.
                elif isinstance(replicants[i], dict):
                    position = replicants[i]
                else:
                    raise Exception(f"Invalid Replicant position: {replicants[i]}")
                # Add a Replicant position.
                replicant_positions.append(position)
                # Occupy the position.
                rooms = self._occupy_position(position=position)
        # Add the Replicants.
        for i, replicant_position in enumerate(replicant_positions):
            replicant = ReplicantTransportChallenge(replicant_id=i,
                                                    state=self.state,
                                                    position=replicant_position,
                                                    image_frequency=self._image_frequency,
                                                    target_framerate=self._target_framerate,
                                                    enable_collision_detection=self.enable_collision_detection)
            self.replicants[replicant.replicant_id] = replicant
            self.add_ons.append(replicant)
        commands = []
        # Add containers.
        room_indices = list(rooms.keys())
        self._rng.shuffle(room_indices)
        if container_room_index is None:
            container_room = rooms[room_indices[0]]
        else:
            container_room = rooms[container_room_index]
        container_room_position_indices = np.arange(0, len(container_room), dtype=int)
        self._rng.shuffle(container_room_position_indices)
        for i in range(num_containers):
            if i >= len(container_room):
                break
            container_id = Controller.get_unique_id()
            # Remember the container.
            self.state.container_ids.append(container_id)
            # Get the container's position.
            position = TDWUtils.vector3_to_array(container_room[container_room_position_indices[i]])
            position += self._rng.uniform(-0.05, 0.05, 3)
            position[1] = 0
            position_v3 = TDWUtils.array_to_vector3(position)
            # Get the container name.
            container_name: str = self._container_names[self._rng.randint(0, len(self._container_names))]
            # Add the container. Use custom physics parameters to set high friction and low bounciness.
            commands.extend(Controller.get_add_physics_object(model_name=container_name,
                                                              object_id=container_id,
                                                              position=position_v3,
                                                              rotation={"x": 0,
                                                                        "y": float(self._rng.uniform(0, 360)),
                                                                        "z": 0},
                                                              default_physics_values=False,
                                                              scale_mass=False,
                                                              mass=5,
                                                              dynamic_friction=0.8,
                                                              static_friction=0.8,
                                                              bounciness=0.1))
            # Update the occupancy map.
            rooms = self._occupy_position(position=position_v3)
        # Add target objects.
        if target_objects_room_index is None:
            target_objects_room = rooms[room_indices[1]]
        else:
            target_objects_room = rooms[target_objects_room_index]
        target_object_room_position_indices = np.arange(0, len(target_objects_room), dtype=int)
        self._rng.shuffle(target_object_room_position_indices)
        for i in range(num_target_objects):
            if i >= len(target_objects_room):
                break
            target_object_id = Controller.get_unique_id()
            # Remember the target object.
            self.state.target_object_ids.append(target_object_id)
            # Get the target object's position.
            position = TDWUtils.vector3_to_array(target_objects_room[target_object_room_position_indices[i]])
            position += self._rng.uniform(-0.05, 0.05, 3)
            position[1] = 0
            position_v3 = TDWUtils.array_to_vector3(position)
            # Get the target object name.
            target_object_name: str = self._target_object_names[self._rng.randint(0, len(self._target_object_names))]
            # Get the target object scale.
            target_object_scale: float = self._target_objects_names_and_scales[target_object_name]
            # Add the target object.
            commands.extend(Controller.get_add_physics_object(model_name=target_object_name,
                                                              object_id=target_object_id,
                                                              position=position_v3,
                                                              rotation={"x": 0,
                                                                        "y": float(self._rng.uniform(0, 360)),
                                                                        "z": 0},
                                                              scale_factor={"x": target_object_scale,
                                                                            "y": target_object_scale,
                                                                            "z": target_object_scale},
                                                              scale_mass=False,
                                                              default_physics_values=False,
                                                              mass=TransportChallenge.TARGET_OBJECT_MASS,
                                                              dynamic_friction=0.8,
                                                              static_friction=0.8,
                                                              bounciness=0.1))
            # Set a random visual material for each target object.
            visual_material: str = self._rng.choice(self._target_object_visual_materials)
            substructure = Controller.MODEL_LIBRARIANS["models_core.json"].get_record(target_object_name).substructure
            visual_material_commands = TDWUtils.set_visual_material(substructure=substructure,
                                                                    material=visual_material,
                                                                    object_id=target_object_id,
                                                                    c=self,
                                                                    quality="low")
            commands.extend(visual_material_commands)
        # Choose a goal room.
        rooms: Dict[int, List[Dict[str, float]]] = self._get_rooms_map(communicate=False)
        if goal_room_index is None:
            room_indices: List[int] = list(rooms.keys())
            goal_room_index = room_indices[self._rng.randint(0, len(room_indices))]
        # Convert the list of room points to a numpy array.
        room_points: np.ndarray = np.array([TDWUtils.vector3_to_array(p) for p in rooms[room_indices[goal_room_index]]])
        # Get the centroid of the room.
        room_centroid: np.ndarray = np.mean(room_points[:, -3:], axis=0)
        # noinspection PyArgumentList
        nearest_index: int = cKDTree(room_points).query(room_centroid, k=1)[1]
        # Set the goal position.
        self.goal_position = room_points[nearest_index]
        # Add a challenge state and object manager.
        self.object_manager.reset()
        self.add_ons.extend([self.state, self.object_manager])
        # Initialize the scene, adding the containers, target objects, and Replicants.
        self.communicate(commands)
        self.communicate([])
        commands.clear()
        # Make all high-mass objects kinematic.
        for object_id in self.object_manager.objects_static:
            o = self.object_manager.objects_static[object_id]
            if o.mass >= TransportChallenge.KINEMATIC_MASS and not o.kinematic:
                commands.append({"$type": "set_kinematic_state",
                                 "id": object_id,
                                 "is_kinematic": True,
                                 "use_gravity": False})
                self.object_manager.objects_static[object_id].kinematic = True
        for replicant_id in self.replicants:
            # Set pass masks.
            commands.append({"$type": "set_pass_masks",
                             "pass_masks": self._image_passes,
                             "avatar_id": self.replicants[replicant_id].static.avatar_id})
            # Ignore collisions with target objects.
            self.replicants[replicant_id].collision_detection.exclude_objects.extend(self.state.target_object_ids)
        # Add a NavMesh.
        nav_mesh_exclude_objects = list(self.replicants.keys())
        nav_mesh_exclude_objects.extend(self.state.target_object_ids)
        nav_mesh = NavMesh(exclude_objects=nav_mesh_exclude_objects)
        self.add_ons.append(nav_mesh)
        # Send the commands.
        self.communicate(commands)
        # Reset the heads.
        for replicant_id in self.replicants:
            self.replicants[replicant_id].reset_head(scale_duration=Globals.SCALE_IK_DURATION)
        reset_heads = False
        while not reset_heads:
            self.communicate([])
            reset_heads = True
            for replicant_id in self.replicants:
                if self.replicants[replicant_id].action.status == ActionStatus.ongoing:
                    reset_heads = False
                    break

    def _get_rooms_map(self, communicate: bool) -> Dict[int, List[Dict[str, float]]]:
        # Generate a new occupancy map and request scene regions data.
        if communicate:
            self.occupancy_map.generate()
            resp = self.communicate([{"$type": "send_scene_regions"}])
            self._scene_bounds = SceneBounds(resp=resp)
        rooms: Dict[int, List[Dict[str, float]]] = dict()
        for ix in range(self.occupancy_map.occupancy_map.shape[0]):
            for iz in range(self.occupancy_map.occupancy_map.shape[1]):
                # Ignore non-free positions.
                if self.occupancy_map.occupancy_map[ix][iz] != 0:
                    continue
                # Find the room that this position is in.
                p = self.occupancy_map.positions[ix][iz]
                for i, region in enumerate(self._scene_bounds.regions):
                    if region.is_inside(p[0], p[1]):
                        if i not in rooms:
                            rooms[i] = list()
                        rooms[i].append({"x": float(p[0]),
                                         "y": 0,
                                         "z": float(p[1])})
                        break
        return rooms

    def _occupy_position(self, position: Dict[str, float]) -> Dict[int, List[Dict[str, float]]]:
        origin = TDWUtils.vector3_to_array(position)
        for ix in range(self.occupancy_map.occupancy_map.shape[0]):
            for iz in range(self.occupancy_map.occupancy_map.shape[1]):
                # Ignore non-free positions.
                if self.occupancy_map.occupancy_map[ix][iz] != 0:
                    continue
                # Find the room that this position is in.
                p2 = self.occupancy_map.positions[ix][iz]
                p3 = np.array([p2[0], 0, p2[1]])
                if np.linalg.norm(origin - p3) <= 0.5:
                    self.occupancy_map.occupancy_map[ix][iz] = 1
        return self._get_rooms_map(communicate=False)