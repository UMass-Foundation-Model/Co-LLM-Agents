import json
from tdw.scene_data.scene_bounds import SceneBounds

def shift(bounds, id = 0):
    eps = 1e-3
    y = bounds.top[1]
    z1 = bounds.back[-1]+eps
    z2 = bounds.front[-1]-eps
    x1 = bounds.left[0]+eps
    x2 = bounds.right[0]-eps
    from random import random
    z = z1 + (0.2 + 0.6 * (id % 2) + 0.1 * random()) * (z2 - z1)
    x = x1 + (0.2 + 0.6 * (id // 2 % 2) + 0.1 * random()) * (x2 - x1)
    return x, y, z

def belongs_to_which_room(x: float, z: float, scene_bounds: SceneBounds):
    '''判断一个坐标(x, z)位于给定的FloorPlan场景中的哪个房间范围内。若不在任何一个房间内则返回-1。
    
    ### Params:
    
    x, z: 坐标
    
    scene_bounds: 场景的房间边界信息，通过
    ```
    resp = controller.communicate([{"$type": "send_scene_regions"}])
    scene_bounds = SceneBounds(resp=resp)
    ```
    获取。
    '''
    for i, region in enumerate(scene_bounds.regions):
        if region.is_inside(x, z):
            return i
    return -1


with open("./dataset/room_types.json") as f:
    room_functionals = json.load(f)


def get_total_rooms(floorplan_scene: str) -> int:
    '''根据FloorPlan的scene名称获取房间总数。
    
    ### 示例：
    ```
    >>> get_total_rooms("2b")
    8
    ```
    '''
    return len(room_functionals[floorplan_scene[0]][0])
    
    
def get_room_functional_by_id(floorplan_scene: str, floorplan_layout: int, room_id: int) -> str:
    '''根据FloorPlan的scene名称, layout, room_id获取房间类型（功能）。
    
    ### 示例：
    ```
    >>> get_room_functional_by_id("2b", 1, 1)
    Livingroom
    ```
    '''
    return room_functionals[floorplan_scene[0]][floorplan_layout][room_id]