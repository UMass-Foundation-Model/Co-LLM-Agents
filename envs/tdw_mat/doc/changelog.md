## 0.2.2

- Required version of TDW is 1.11.15. There may be very minor changes to the Replicant's behavior.

## 0.2.1

- Required version of TDW is 1.11.8.3.

## 0.2.0

- Added changelog.
- Required version of TDW is 1.11.8.
- Added the goal zone to the `TransportChallenge` controller:
  - Added: `TransportChallenge.GOAL_ZONE_RADIUS`.
  - Added: `self.goal_position`. The `TransportChallenge` controller sets this at the start of a new trial.
  - Added optional parameter `goal_room_index` to `self.start_floorplan_trial()`.
  - Added documentation for goal zones to the README.
- Added missing documentation for `self.occupancy_map` and `self.object_manager` in `TransportChallenge`.