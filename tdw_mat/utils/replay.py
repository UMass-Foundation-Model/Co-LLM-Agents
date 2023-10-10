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
from tdw.add_ons.log_playback import LogPlayback
import os

c = Controller(port=10001)
log_playback = LogPlayback()
c.add_ons.append(log_playback)
# Load the commands.
log_playback.load(path="results/llm_llm_comm_seed_0/10/action_log_v1.log")
# Play back each list of commands.
frame = 0
while len(log_playback.playback) > 0:
    data = c.communicate([])
    for i in range(len(data) - 1):
        r_id = OutputData.get_data_type_id(data[i])
        if r_id == 'imag':
            images = Images(data[i])
            if images.get_avatar_id() == "a" and (frame) % 1 == 0:
                TDWUtils.save_images(images=images, filename= f"{frame:05d}", output_directory = os.path.join('replay/replay_10_arxiv', 'images'))
    frame += 1
    print(frame)
c.communicate({"$type": "terminate"})