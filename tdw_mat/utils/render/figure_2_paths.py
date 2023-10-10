from typing import List, Dict, Tuple
import numpy as np
import re
from pathlib import Path
from tdw.controller import Controller
from tdw.add_ons.log_playback import LogPlayback
from tdw.output_data import OutputData, Replicants, ScreenPosition
from tdw.quaternion_utils import QuaternionUtils
from tdw.tdw_utils import TDWUtils

output_directory = Path("figure_2").resolve()
log_path = Path("figure_2.log")
# "Fix" the log path.
log_text = log_path.read_text(encoding="utf-8")
# Set the screen size to very small.
log_text = re.sub(r'({"\$type": "set_screen_size", "width": (.*?), "height": (.*?)\})', '{"$type": "set_screen_size", "width": 64, "height": 64}', log_text)
# Never send images.
# log_text = re.sub(r'({"\$type": "send_images", "frequency": "(.*?)"})', '{"$type": "send_images", "frequency": "never"}', log_text)
# Replace the skybox.
log_text = re.sub(r'({"\$type": "add_hdri_skybox", (.*?)})', r'{"$type": "add_hdri_skybox", "name": "sky_white", "url": "https://tdw-public.s3.amazonaws.com/hdri_skyboxes/linux/2019.1/sky_white", "exposure": 2, "initial_skybox_rotation": 0, "sun_elevation": 90, "sun_initial_angle": 0, "sun_intensity": 1.25}', log_text)
# Move the camera.
log_text = log_text.replace('{"$type": "teleport_avatar_to", "position": {"x": 0, "y": 20, "z": 0}, "avatar_id": "a"}', '{"$type": "teleport_avatar_to", "position": {"x": 0, "y": 25, "z": 0}, "avatar_id": "a"}')
# Write the fixed version.
log_path = Path("figure_2_fixed.log")
log_path.write_text(log_text)
player = LogPlayback()
player.load(path=log_path)
c = Controller()
player = LogPlayback()
player.load(path=log_path)
c.add_ons.append(player)
replicant_0_path: List[np.ndarray] = list()
replicant_1_path: List[np.ndarray] = list()
replicant_0_rotations: List[float] = list()
replicant_1_rotations: List[float] = list()
# Remove the last frame (terminate).
player.playback = player.playback[:-1]
while len(player.playback) > 0:
    resp = c.communicate([])
    for i in range(len(resp) - 1):
        r_id = OutputData.get_data_type_id(resp[i])
        if r_id == "repl":
            repl = Replicants(resp[i])
            frame: Dict[int, Tuple[np.ndarray, np.ndarray]] = dict()
            for j in range(repl.get_num()):
                replicant_id = repl.get_id(j)
                position = repl.get_position(j)
                rotation = QuaternionUtils.quaternion_to_euler_angles(repl.get_rotation(j))[1]
                if replicant_id == 0:
                    replicant_0_rotations.append(rotation)
                    replicant_0_path.append(position)
                else:
                    replicant_1_rotations.append(rotation)
                    replicant_1_path.append(position)
np.save(str(output_directory.joinpath("replicant_0_path").resolve()), np.array(replicant_0_path))
np.save(str(output_directory.joinpath("replicant_1_path").resolve()), np.array(replicant_1_path))
np.save(str(output_directory.joinpath("replicant_0_rotations").resolve()), np.array(replicant_0_rotations))
np.save(str(output_directory.joinpath("replicant_1_rotations").resolve()), np.array(replicant_1_rotations))

# Get the screen positions.
positions = [TDWUtils.array_to_vector3(np.array([p[0], 0.1, p[2]])) for p in replicant_0_path]
positions.extend([TDWUtils.array_to_vector3(np.array([p[0], 0.1, p[2]])) for p in replicant_1_path])
avatar_ids = ["a" for _ in range(len(replicant_0_path))]
avatar_ids.extend(["a" for _ in range(len(replicant_1_path))])
position_ids = list(range(len(replicant_0_path)))
id_offset = 5000
position_ids.extend([i + id_offset for i in range(len(replicant_1_path))])
c.communicate([{"$type": "set_screen_size", "width": 1920, "height": 1080}])
resp = c.communicate([{"$type": "send_screen_positions", "positions": positions, "ids": avatar_ids, "position_ids": position_ids}])
screen_positions = np.zeros(shape=(2, len(replicant_0_path), 2))
for i in range(len(resp) - 1):
    r_id = OutputData.get_data_type_id(resp[i])
    if r_id == "scre":
        screen_position = ScreenPosition(resp[i])
        position = screen_position.get_screen()
        position_id = screen_position.get_id()
        if position_id < id_offset:
            replicant_id = 0
        else:
            replicant_id = 1
            position_id = position_id - id_offset
        screen_positions[replicant_id][position_id] = np.array([position[0], position[1]])
# Save the screen positions.
np.save(str(output_directory.joinpath("screen_positions").resolve()), screen_positions)
c.communicate({"$type": "terminate"})