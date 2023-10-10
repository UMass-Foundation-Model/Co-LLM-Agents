from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.replicant.action_status import ActionStatus
from tdw.replicant.arm import Arm
from transport_challenge_multi_agent.transport_challenge import TransportChallenge

"""
A multi agent simulation in a box room scene.

Both Replicants move to containers and pick them up.
"""


def do_actions():
    doing = True
    while doing:
        doing = False
        for replicant_id in c.replicants:
            if c.replicants[replicant_id].action.status == ActionStatus.ongoing:
                doing = True
                break
        if doing:
            c.communicate([])
    c.communicate([])


c = TransportChallenge()
c.start_box_room_trial(size=(6, 6),
                       num_containers=1,
                       num_target_objects=1,
                       replicants=2,
                       random_seed=0)
camera = ThirdPersonCamera(avatar_id="a",
                           position={"x": -1.82, "y": 2.6, "z": 0.87},
                           look_at=c.replicants[0].replicant_id)
c.add_ons.append(camera)
c.replicants[0].move_to_object(target=c.state.container_ids[0])
c.replicants[1].move_to_object(target=c.state.container_ids[0])
do_actions()
c.replicants[0].pick_up(target=c.state.container_ids[0])
c.replicants[1].pick_up(target=c.state.container_ids[0])
do_actions()
# One of these Replicants failed to grasp. Which one is it?
if Arm.left in c.replicants[0].dynamic.held_objects:
    holding = 0
    not_holding = 1
else:
    holding = 1
    not_holding = 0
# The Replicant that isn't holding the container will try to pick it up and immediately fail.
c.replicants[not_holding].pick_up(target=c.state.container_ids[0])
do_actions()
assert c.replicants[not_holding].action.status == ActionStatus.cannot_grasp, c.replicants[not_holding].action.status
# The Replicant that is holding the container will try to pick it up and fail.
for arm in [Arm.left, Arm.right]:
    c.replicants[holding].pick_up(target=c.state.container_ids[0])
    do_actions()
    assert c.replicants[holding].action.status == ActionStatus.cannot_grasp, c.replicants[holding].action.status
# Move backward.
c.replicants[not_holding].move_backward()
do_actions()
# Both Replicants will try to put an object in a container and fail.
for i in range(len(c.replicants)):
    c.replicants[i].put_in()
do_actions()
for i in range(len(c.replicants)):
    assert c.replicants[i].action.status != ActionStatus.success, c.replicants[i].action.status
# End.
c.communicate({"$type": "terminate"})
