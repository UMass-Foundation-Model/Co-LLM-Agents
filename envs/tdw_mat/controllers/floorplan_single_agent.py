from tdw.replicant.action_status import ActionStatus
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from transport_challenge_multi_agent.transport_challenge import TransportChallenge


"""
A single agent simulation in a floorplan scene.

The Replicant navigates to a container and picks it up.
"""


c = TransportChallenge(screen_width=1280, screen_height=720)
c.start_floorplan_trial(scene="2a", layout=0, replicants=[{"x": 6.7, "y": 0, "z": -3.46}],
                        num_containers=4, num_target_objects=8, random_seed=1)
camera = ThirdPersonCamera(avatar_id="a",
                           position={"x": 0, "y": 20, "z": 0},
                           look_at=c.replicants[0].replicant_id)
c.add_ons.append(camera)
c.communicate([{"$type": "set_floorplan_roof",
                "show": False}])
c.replicants[0].collision_detection.avoid = False
c.replicants[0].navigate_to(target=c.state.container_ids[0])
while c.replicants[0].action.status == ActionStatus.ongoing:
    c.communicate([])
c.communicate([])
print(c.replicants[0].action.status)
c.replicants[0].pick_up(target=c.state.container_ids[0])
while c.replicants[0].action.status == ActionStatus.ongoing:
    c.communicate([])
c.communicate([])
print(c.replicants[0].action.status)
c.communicate({"$type": "terminate"})
