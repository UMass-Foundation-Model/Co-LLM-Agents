from tdw.replicant.action_status import ActionStatus
from transport_challenge_multi_agent.transport_challenge import TransportChallenge

"""
A single agent simulation in an box room scene. The Replicant looks up and down and then resets its head.
"""


def do_action():
    while c.replicants[0].action.status == ActionStatus.ongoing:
        c.communicate([])
    c.communicate([])


c = TransportChallenge()
c.start_box_room_trial(size=(6, 6),
                       num_containers=0,
                       num_target_objects=0,
                       replicants=[{"x": 0, "y": 0, "z": 0}],
                       random_seed=1)
c.communicate([])
c.replicants[0].look_up(15)
do_action()
c.replicants[0].look_down(30)
do_action()
c.replicants[0].reset_head()
do_action()
c.communicate({"$type": "terminate"})
