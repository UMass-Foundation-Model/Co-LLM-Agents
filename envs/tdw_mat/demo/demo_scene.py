from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.logger import Logger
from tdw.output_data import OutputData, SegmentationColors, FieldOfView, Images
from transport_challenge_multi_agent.replicant_transport_challenge import ReplicantTransportChallenge
from transport_challenge_multi_agent.challenge_state import ChallengeState
from tdw.replicant.arm import Arm
from tdw.replicant.action_status import ActionStatus
from os import chdir
from subprocess import call
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.backend.paths import EXAMPLE_CONTROLLER_OUTPUT_PATH
import time
c = Controller(port=1072)
#logger = Logger('log.txt')
#c.add_ons.extend([logger])
camera = ThirdPersonCamera(position={"x": -5, "y": 4, "z": 0}, avatar_id="a", look_at={"x": 5, "y": -4, "z": 0})
c.add_ons.extend([camera])
state = ChallengeState()
commands = [TDWUtils.create_empty_room(12, 12),
            {"$type": "set_screen_size",
             "width": 1024,
             "height": 1024}]
object_ids = [Controller.get_unique_id(), Controller.get_unique_id(), Controller.get_unique_id(), Controller.get_unique_id(), Controller.get_unique_id(), Controller.get_unique_id()]
commands.extend(Controller.get_add_physics_object(model_name="pencil_all",
                                                  position={"x": 0, "y": 0.15, "z": -0.5},
                                                  rotation={"x": 0,
                                                    "y": 234,
                                                    "z": 0},
                                                  scale_factor={"x": 1.2,
                                                        "y": 1.2,
                                                        "z": 1.2},
                                                  object_id=object_ids[0]))
commands.extend(Controller.get_add_physics_object(model_name="b05_calculator",
                                                  position={"x": 0, "y": 0.15, "z": -1},
                                                  rotation={"x": 0,
                                                    "y": 234,
                                                    "z": 0},
                                                  scale_factor={"x": 2,
                                                        "y": 2,
                                                        "z": 2},
                                                  object_id=object_ids[1]))
commands.extend(Controller.get_add_physics_object(model_name="mouse_02_vray",
                                                  position={"x": -0.5, "y": 0.15, "z": -1},
                                                  rotation={"x": 0,
                                                    "y": 234,
                                                    "z": 0},
                                                  scale_factor={"x": 1.5,
                                                        "y": 1.5,
                                                        "z": 1.5},
                                                  object_id=object_ids[2]))
commands.extend(Controller.get_add_physics_object(model_name="b04_orange_00",
                                                  position={"x": -1, "y": 0.15, "z": -0.5},
                                                  rotation={"x": 0,
                                                    "y": 234,
                                                    "z": 0},
                                                  scale_factor={"x": 1,
                                                        "y": 1,
                                                        "z": 1},
                                                  object_id=object_ids[3]))
commands.extend(Controller.get_add_physics_object(model_name="b03_burger",
                                                  position={"x": -1, "y": 0.15, "z": -1},
                                                  rotation={"x": 0,
                                                    "y": 234,
                                                    "z": 0},
                                                  scale_factor={"x": 1.2,
                                                        "y": 1.2,
                                                        "z": 1.2},
                                                  object_id=object_ids[4]))
commands.extend(Controller.get_add_physics_object(model_name="b03_loafbread",
                                                  position={"x": -1.5, "y": 0.15, "z": -1},
                                                  rotation={"x": 0,
                                                    "y": 234,
                                                    "z": 0},
                                                  scale_factor={"x": 0.7,
                                                        "y": 0.7,
                                                        "z": 0.7},
                                                  object_id=object_ids[5]))

state.target_object_ids += object_ids
container_id = Controller.get_unique_id()
commands.extend(Controller.get_add_physics_object(model_name="teatray",
                                                  position={"x": 0, "y": 0.2, "z": 0.5},
                                                  scale_factor={"x": 1.25,
                                                        "y": 1.25,
                                                        "z": 1.25},
                                                  object_id=container_id))
state.container_ids.append(container_id)
replicant = ReplicantTransportChallenge(replicant_id=0,
                                                    state=state,
                                                    position={"x": 1, "y": 0, "z": 0},
                                                    enable_collision_detection=False)
for x in object_ids:
    replicant.collision_detection.exclude_objects.append(x)
for x in state.container_ids:
    replicant.collision_detection.exclude_objects.append(x)
c.add_ons.extend([replicant])
c.communicate(commands)
action_buffer = []
action_buffer.append({'type': 'reach_for', 'object': container_id})
action_buffer.append({'type': 'grasp', 'arm': Arm.left, 'object': container_id})
action_buffer.append({'type': 'reach_for', 'object': object_ids[0]})
action_buffer.append({'type': 'grasp', 'arm': Arm.right, 'object': object_ids[0]})
action_buffer.append({'type': 'put_in'})
action_buffer.append({'type': 'reach_for', 'object': object_ids[1]})
action_buffer.append({'type': 'grasp', 'arm': Arm.right, 'object': object_ids[1]})
action_buffer.append({'type': 'put_in'})
action_buffer.append({'type': 'reach_for', 'object': object_ids[2]})
action_buffer.append({'type': 'grasp', 'arm': Arm.right, 'object': object_ids[2]})
action_buffer.append({'type': 'put_in'})
action_buffer.append({'type': 'reach_for', 'object': object_ids[3]})
action_buffer.append({'type': 'grasp', 'arm': Arm.right, 'object': object_ids[3]})
action_buffer.append({'type': 'put_in'})
action_buffer.append({'type': 'reach_for', 'object': object_ids[4]})
action_buffer.append({'type': 'grasp', 'arm': Arm.right, 'object': object_ids[4]})
action_buffer.append({'type': 'put_in'})
action_buffer.append({'type': 'reach_for', 'object': object_ids[5]})
action_buffer.append({'type': 'grasp', 'arm': Arm.right, 'object': object_ids[5]})
action_buffer.append({'type': 'put_in'})
num_frames = 0
finish = False
while not finish: # continue until any agent's action finishes
    if replicant.action.status != ActionStatus.ongoing and len(action_buffer) == 0:
        finish = True
    elif replicant.action.status != ActionStatus.ongoing:
        curr_action = action_buffer.pop(0)
        if curr_action['type'] == 'move_forward':       # move forward 0.5m
            replicant.move_forward()
        elif curr_action['type'] == 'turn_left':     # turn left by 15 degree
            replicant.turn_by(angle = -15)
        elif curr_action['type'] == 'turn_right':     # turn right by 15 degree
            replicant.turn_by(angle = 15)
        elif curr_action['type'] == 'reach_for':     # go to and grasp object with arm
            replicant.move_to_object(int(curr_action["object"]))
        elif curr_action['type'] == 'grasp':
            replicant.grasp(int(curr_action["object"]), curr_action["arm"], relative_to_hand = False, axis = "yaw")
        elif curr_action["type"] == 'put_in':      # put in container
            replicant.put_in()
        elif curr_action["type"] == 'drop':      # drop held object in arm
            replicant.drop(curr_action['arm'])
    if finish: break
    time.sleep(0.1)
    data = c.communicate([])
    for i in range(len(data) - 1):
        r_id = OutputData.get_data_type_id(data[i])
        if r_id == 'imag':
            images = Images(data[i])
            if images.get_avatar_id() == "a":
                TDWUtils.save_images(images=images, filename= f"{num_frames}", output_directory = 'demo/demo_images')
    num_frames += 1
c.communicate({"$type": "terminate"})
