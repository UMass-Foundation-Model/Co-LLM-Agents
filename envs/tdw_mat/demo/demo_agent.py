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
import os
c = Controller(port=1072)
#logger = Logger('log.txt')
#c.add_ons.extend([logger])
camera = ThirdPersonCamera(position={"x": 2, "y": 2, "z": 2}, avatar_id="a", look_at={"x": -2, "y": -1, "z": -2})
c.add_ons.extend([camera])
''''''
state = ChallengeState()
commands = [TDWUtils.create_empty_room(12, 12),
            {"$type": "set_screen_size",
             "width": 1024,
             "height": 1024}]
replicant = ReplicantTransportChallenge(replicant_id=0,
                                                    state=state,
                                                    position={"x": 0, "y": 0, "z": 0},
                                                    enable_collision_detection=False)
c.add_ons.extend([replicant])
c.communicate([{"$type": "set_screen_size",
             "width": 1024,
             "height": 1024}])
action_buffer = []
num_frames = 0
finish = False
while not finish or num_frames < 10: # continue until any agent's action finishes
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
    time.sleep(0.1)
    data = c.communicate([])
    for i in range(len(data) - 1):
        r_id = OutputData.get_data_type_id(data[i])
        if r_id == 'imag':
            images = Images(data[i])
            if images.get_avatar_id() == "a":
                TDWUtils.save_images(images=images, filename= f"{num_frames}", output_directory = 'demo/')
    num_frames += 1
c.communicate({"$type": "terminate"})
