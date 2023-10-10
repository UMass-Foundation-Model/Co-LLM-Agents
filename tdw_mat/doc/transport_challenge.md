# TransportChallenge

`from transport_challenge_multi_agent.transport_challenge import TransportChallenge`

A subclass of `Controller` for the Transport Challenge. Always use this class instead of `Controller`.

See the README for information regarding output data and scene initialization.

***

## Class Variables

| Variable | Type | Description | Value |
| --- | --- | --- | --- |
| `TARGET_OBJECT_MASS` | float | The mass of each target object. | `0.25` |
| `KINEMATIC_MASS` | float | If an object is has this mass or greater, the object will become kinematic. | `100` |
| `TDW_VERSION` | str | The expected version of TDW. | `"1.11.8"` |
| `GOAL_ZONE_RADIUS` | float | The goal zone is a circle defined by `self.goal_center` and this radius value. | `1` |

***

## Fields

- `replicants` A dictionary of all Replicants in the scene. Key = The Replicant ID. Value = [`ReplicantTransportChallenge`](replicant_transport_challenge.md).

- `state` The `ChallengeState`, which includes container IDs, target object IDs, containment state, and which Replicant is holding which objects.

- `occupancy_map` An [`OccupancyMap`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/navigation/occupancy_maps.md). This is used to place objects and can also be useful for navigation.

- `object_manager` An [`ObjectManager`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/core_concepts/output_data.md#the-objectmanager-add-on) add-on. This is useful for referencing all objects in the scene.

- `goal_position` The challenge is successful when the Replicants move all of the target objects to the the goal zone, which is defined by this position and `TransportChallenge.GOAL_ZONE_RADIUS`. This value is set at the start of a trial.

***

## Functions

#### \_\_init\_\_

**`TransportChallenge()`**

**`TransportChallenge(port=1071, check_version=True, launch_build=True, screen_width=256, screen_height=256, image_frequency=ImageFrequency.once, png=True, image_passes=None, target_framerate=250)`**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| port |  int  | 1071 | The socket port used for communicating with the build. |
| check_version |  bool  | True | If True, the controller will check the version of the build and print the result. |
| launch_build |  bool  | True | If True, automatically launch the build. If one doesn't exist, download and extract the correct version. Set this to False to use your own build, or (if you are a backend developer) to use Unity Editor. |
| screen_width |  int  | 256 | The width of the screen in pixels. |
| screen_height |  int  | 256 | The height of the screen in pixels. |
| image_frequency |  ImageFrequency  | ImageFrequency.once | How often each Replicant will capture an image. `ImageFrequency.once` means once per action, at the end of the action. `ImageFrequency.always` means every communicate() call. `ImageFrequency.never` means never. |
| png |  bool  | True | If True, the image pass from each Replicant will be a lossless png. If False, the image pass from each Replicant will be a lossy jpg. |
| image_passes |  List[str] | None | A list of image passes, such as `["_img"]`. If None, defaults to `["_img", "_id", "_depth"]` (i.e. image, segmentation colors, depth maps). |
| target_framerate |  int  | 250 | The target framerate. It's possible to set a higher target framerate, but doing so can lead to a loss of precision in agent movement. |

#### start_floorplan_trial

**`self.start_floorplan_trial(scene, layout, num_containers, num_target_objects, replicants)`**

**`self.start_floorplan_trial(scene, layout, num_containers, num_target_objects, container_room_index=None, target_objects_room_index=None, goal_room_index=None, replicants, lighting=True, random_seed=None)`**

Start a trial in a floorplan scene.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| scene |  str |  | The scene. Options: "1a", "1b", "1c", "2a", "2b", "2c", "4a", "4b", "4c", "5a", "5b", "5c" |
| layout |  int |  | The layout of objects in the scene. Options: 0, 1, 2. |
| num_containers |  int |  | The number of containers in the scene. |
| num_target_objects |  int |  | The number of target objects in the scene. |
| container_room_index |  int  | None | The index of the room in which containers will be placed. If None, the room is random. |
| target_objects_room_index |  int  | None | The index of the room in which target objects will be placed. If None, the room is random. |
| goal_room_index |  int  | None | The index of the goal room. If None, the room is random. |
| replicants |  Union[int, List[Union[int, np.ndarray, Dict[str, float] |  | An integer or a list. If an integer, this is the number of Replicants in the scene; they will be added at random positions on the occupancy map. If a list, each element can be: An integer (the Replicant will be added *in this room* at a random occupancy map position), a numpy array (a worldspace position), or a dictionary (a worldspace position, e.g. `{"x": 0, "y": 0, "z": 0}`. |
| lighting |  bool  | True | If True, add an HDRI skybox for realistic lighting. The skybox will be random, using the `random_seed` value below. |
| random_seed |  int  | None | The random see used to add containers, target objects, and Replicants, as well as to set the lighting and target object materials. If None, the seed is random. |

#### start_box_room_trial

**`self.start_box_room_trial(size, num_containers, num_target_objects, replicants)`**

**`self.start_box_room_trial(size, num_containers, num_target_objects, replicants, random_seed=None)`**

Start a trial in a simple box room scene.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| size |  Tuple[int, int] |  | The size of the room. |
| num_containers |  int |  | The number of containers in the scene. |
| num_target_objects |  int |  | The number of target objects in the scene. |
| replicants |  Union[int, List[Union[np.ndarray, Dict[str, float] |  | An integer or a list. If an integer, this is the number of Replicants in the scene; they will be added at random positions on the occupancy map. If a list, each element can be: A numpy array (a worldspace position), or a dictionary (a worldspace position, e.g. `{"x": 0, "y": 0, "z": 0}`. |
| random_seed |  int  | None | The random see used to add containers, target objects, and Replicants, as well as to set the lighting and target object materials. If None, the seed is random. |

#### communicate

**`self.communicate(commands)`**

Send commands and receive output data in response.


| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| commands |  Union[dict, List[dict] |  | A list of JSON commands. |

_Returns:_  The output data from the build.

