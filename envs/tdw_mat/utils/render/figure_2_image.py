from pathlib import Path
from tdw.controller import Controller
from tdw.add_ons.log_playback import LogPlayback
from tdw.add_ons.image_capture import ImageCapture

# Load the log file.
log_path = Path("figure_2_fixed.log")
player = LogPlayback()
player.load(log_path)
# Only play back to the designated frame.
frame = 1185
player.playback = player.playback[:frame]
c = Controller()
c.add_ons.append(player)
for i in range(frame):
    c.communicate([])
# Capture an image.
capture = ImageCapture(avatar_ids=["a"], pass_masks=["_img"], path=Path("figure_2"))
c.add_ons.append(capture)
c.communicate({"$type": "set_screen_size", "width": 1920, "height": 1080})
c.communicate({"$type": "terminate"})