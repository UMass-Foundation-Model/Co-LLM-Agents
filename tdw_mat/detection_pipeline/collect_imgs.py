import numpy as np
from tdw.replicant.action_status import ActionStatus
from transport_challenge_multi_agent.transport_challenge import TransportChallenge
from tdw.replicant.image_frequency import ImageFrequency
from tdw.add_ons.image_capture import ImageCapture
from tdw.output_data import OutputData, SegmentationColors, CameraMatrices
import random
import time
import json
import os


def generate_imgs(save_dir, data_prefix, scene, layout, task):
    # TODO: change NavigateTo() to Nav() in test_h_agent
    _start = time.time()
    print("Start ---------------------------------", time.strftime("%H:%M:%S", time.localtime(_start)))
    c = TransportChallenge(
        check_version=True, screen_width=512, screen_height=512,
        image_frequency=ImageFrequency.always, png=False, image_passes=["_img", "_id", "_depth"],
        launch_build=True, new_setting=True, port = 1999)
    print("Build Scene ---------------------------------", time.time() - _start)
    c.start_floorplan_trial(scene=scene, layout=str(layout) + "_1", replicants=1, num_containers=4, num_target_objects=10,
                            random_seed=None, task=task, data_prefix=data_prefix)  # Random Spawn Replicant
    print("FOV -------------------------------", time.time() - _start)
    c.communicate({"$type": "set_field_of_view", "avatar_id" : str(0), "field_of_view" : 90})
    print("Floor -------------------------------", time.time() - _start)
    c.communicate([{"$type": "set_floorplan_roof", "show": False}])
    print("Sky -------------------------------", time.time() - _start)
    c.communicate({"$type": "add_hdri_skybox", "name": "sky_white",
                                 "url": "https://tdw-public.s3.amazonaws.com/hdri_skyboxes/linux/2019.1/sky_white",
                                 "exposure": 2, "initial_skybox_rotation": 0, "sun_elevation": 90,
                                 "sun_initial_angle": 0, "sun_intensity": 1.25})
    #print("FOV -------------------------------", time.time() - _start)
    #c.communicate({"$type": "set_field_of_view", "avatar_id": str(c.replicants[0]), "field_of_view": 90})

    print("Setting -------------------------------", time.time() - _start)
    capture = ImageCapture(avatar_ids=[c.replicants[0].static.avatar_id], path=save_dir,
                           pass_masks=["_img", "_depth", "_id"], png=False)
    c.add_ons.extend([capture])
    c.replicants[0].collision_detection.avoid = False
    obj_ids = save_seg_color_dict(c, data_prefix, save_dir)
    print(time.time() - _start)

    print("Move ----------------------------------", time.time() - _start)
    random_sample(c, obj_ids, save_dir)
    c.communicate({"$type": "terminate"})


def random_sample(c, obj_ids, save_dir):
    rooms = list(c._get_rooms_map(communicate=True).values())
    print(len(rooms))
    # rooms = c._scene_bounds.regions
    random.shuffle(rooms)
    replicant = c.replicants[0]
    pre_pos = None

    with open(os.path.join(save_dir, "logger.txt"), "w") as f:
        for k, region in enumerate(rooms):
            f.write(f"Room: {k}\n")
            step_count = 0
            try_count = 0
            while (step_count <= 500) and (try_count <= 1000):
                # pos = {"x": random.uniform(region.x_min, region.x_max), "y": 0,
                #        "z": random.uniform(region.z_min, region.z_max)}
                pos = random.choice(region)
                replicant.navigate_to(target=pos)
                while replicant.action.status in [ActionStatus.ongoing]:
                    c.communicate([])
                    try_count += 1
                    cur_pos = replicant.dynamic.transform.position
                    if not np.all(pre_pos == cur_pos):
                        step_count += 1
                        pre_pos = cur_pos
                c.communicate([])
                print(replicant.action.status, step_count, try_count)
                f.write(f"{step_count}: {replicant.action.status}\n")

        # f.write(f"Objects\n")
        # step_count = 0
        # try_count = 0
        # while (step_count <= 2000) and (try_count <= 4000):
        #     _id = random.choice(obj_ids)
        #     replicant.navigate_to(target=_id)
        #     while replicant.action.status in [ActionStatus.ongoing]:
        #         c.communicate([])
        #         try_count += 1
        #         cur_pos = replicant.dynamic.transform.position
        #         if not np.all(pre_pos == cur_pos):
        #             step_count += 1
        #             pre_pos = cur_pos
        #     c.communicate([])
        #     print(replicant.action.status, step_count, try_count)
        #     f.write(f"{step_count}: {replicant.action.status}\n")


def save_seg_color_dict(c, data_prefix, save_dir):
    with open(os.path.join(data_prefix, "name_map.json")) as json_file:
        perception_obj_list = json.load(json_file).keys()

    color_obj_map = dict()  # color -> object idx
    obj_ids = []
    response = c.communicate({"$type": "send_segmentation_colors", "frequency": "once"})
    byte_array = filter(lambda x: OutputData.get_data_type_id(x) == "segm", response).__next__()
    seg = SegmentationColors(byte_array)
    for i in range(seg.get_num()):
        _obj = seg.get_object_name(i).lower()
        _color = seg.get_object_color(i)
        _id = seg.get_object_id(i)

        if _obj in perception_obj_list:
            color_obj_map[str(_color)] = _obj
            obj_ids.append(_id)
        elif "bed" == seg.get_object_category(i).lower():
            color_obj_map[str(_color)] = "bed"
            obj_ids.append(_id)

    with open(os.path.join(save_dir, "color_obj_map.json"), "w") as f:
        f.write(json.dumps(color_obj_map))
    return obj_ids


if __name__ == "__main__":
    save_dir = "./captured_imgs_fov90"
    task = "food"
    layout_type = [0, 1, 2]

    train_dataset = "dataset/dataset_train"
    training_scene_list = ["1a", "4a"]
    for i in [8, 9]:
        for layout in layout_type:
            for scene in training_scene_list:
                _dir = os.path.join(save_dir, f"{scene}_{layout}_{i}")
                print(_dir)
                generate_imgs(_dir, train_dataset, scene, layout, task)

    # test_dataset = "dataset/dataset_test"
    # test_scene_list = ["2a", "5a"]
    # for i in range(4):
    #     for layout in layout_type:
    #         for scene in test_scene_list:
    #             _dir = os.path.join(save_dir, f"{scene}_{layout}_{i}")
    #             print(_dir)
    #             generate_imgs(_dir, test_dataset, scene, layout, task)
