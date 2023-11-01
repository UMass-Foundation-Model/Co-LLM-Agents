from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.replicant.action_status import ActionStatus
from transport_challenge_multi_agent.transport_challenge import TransportChallenge

"""
A single agent simulation in an box room scene.

The Replicant picks up a container. Then it goes from object to object, placing each one in the container.
"""


def do_action():
    while c.replicants[0].action.status == ActionStatus.ongoing:
        c.communicate([])
    c.communicate([])


c = TransportChallenge()
c.start_box_room_trial(size=(6, 6),
                       num_containers=1,
                       num_target_objects=5,
                       replicants=[{"x": 0, "y": 0, "z": 0}],
                       random_seed=1)
camera = ThirdPersonCamera(avatar_id="a",
                           position={"x": -1.82, "y": 2.6, "z": 0.87},
                           look_at=c.replicants[0].replicant_id)
c.add_ons.append(camera)
c.replicants[0].move_to_object(target=c.state.container_ids[0])
do_action()
c.replicants[0].pick_up(target=c.state.container_ids[0])
do_action()
for target_object_id in c.state.target_object_ids:
    c.replicants[0].move_to_object(target=target_object_id)
    do_action()
    c.replicants[0].pick_up(target=target_object_id)
    do_action()
    c.replicants[0].put_in()
    do_action()
c.communicate({"$type": "terminate"})
