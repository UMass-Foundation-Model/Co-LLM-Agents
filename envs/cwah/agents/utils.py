import open3d as o3d
import numpy as np
import cv2
import copy
import re

def save_pcd_from_point_array(pos, col = None, filename = 'temp.ply'):
    print(filename)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos)
    if col is not None:
        pcd.colors = o3d.utility.Vector3dVector(col)
    o3d.io.write_point_cloud(filename, pcd)

def read_pcd_from_point_array(pos, col = None)-> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos)
    if col is not None:
        pcd.colors = o3d.utility.Vector3dVector(col)
    return pcd

def inverse_rot(currrot):
    new_matrix = np.zeros((4,4))
    new_matrix[:3, :3] = currrot[:3,:3].transpose()
    new_matrix[-1,-1] = 1
    new_matrix[:-1, -1] = np.matmul(currrot[:3,:3].transpose(), -currrot[:-1, -1])
    return new_matrix

def replace_name(names):
    new_names = []
    for name in names:
        new_name = name.replace('coffeetable', 'coffee table')
        new_names.append(new_name)
    return new_names

def save_navigation_visual_map(grid_info, filename):
    h, w = grid_info.shape
    visualize = np.zeros((h, w, 4))
    for x in range(h):
        for y in range(w):
            if (grid_info[x, y] == 4):
                visualize[x, y, 1:] = [1, 1, 1]
            elif (grid_info[x, y] == 5):
                visualize[x, y, 1:] = [1, 1, 0]
            else: 
                visualize[x, y, grid_info[x, y]] = 1
    cv2.imwrite(filename, visualize[:, :, 1:] * 255)

def image2coords(img, depth, camera_info, far_away_remove = True, remove_threshold = 10, clip = None, mask = None):
    r'''
        for a (img, depth) input, return a list of points, rgb: [x, y, z], [g ,b, r]
        clip: [x_min, y_min, x_max, y_max]
        return points position, points color, points image location, camera position
        #TODO: check whether it works in ws != hs cases
    '''
    if (len(depth.shape) > 2): depth = depth[:, :, 0]
    if (len(img.shape) > 2): (hs, ws, _) = img.shape
    else: (hs, ws) = img.shape
#    if (hs > ws):
#        delt = (hs - ws) / 2
#        img = img[:, int(delt):int(delt + ws)]
#        depth = depth[:, int(delt):int(delt + ws)]
#    if (len(img.shape) > 2): (ws, hs, _) = img.shape
#    else: (ws, hs) = img.shape
    # naspect = float(ws/hs)
    # aspect = camera_info['aspect']
    
    w = np.arange(ws)
    h = np.arange(hs)
    projection = np.array(camera_info['projection_matrix']).reshape((4,4)).transpose()
    view = np.array(camera_info['world_to_camera_matrix']).reshape((4,4)).transpose()

    # Build inverse projection
    inv_view = inverse_rot(view)
    # img = img.reshape(-1,3)
    # col = img

    xv, yv = np.meshgrid(w, h)
    ori_x, ori_y = np.meshgrid(w, h) # check it
    if type(clip) != type(None):
        clip = [int(clip[0]), int(clip[1]), int(clip[2]), int(clip[3])]
        xv = xv[clip[1]:clip[3], clip[0]:clip[2]]
        yv = yv[clip[1]:clip[3], clip[0]:clip[2]]
        ori_x = ori_x[clip[1]:clip[3], clip[0]:clip[2]]
        ori_y = ori_y[clip[1]:clip[3], clip[0]:clip[2]]
        img = img[clip[1]:clip[3], clip[0]:clip[2]]
        depth = depth[clip[1]:clip[3], clip[0]:clip[2]]
        if type(mask) != type(None):
            mask = mask[clip[1]:clip[3], clip[0]:clip[2]]

    # npoint = ws * hs
    # Normalize image coordinates to -1, 1
    # -1 to 1
    # xp = xv.reshape((-1))
    # yp = yv.reshape((-1))

    x = xv.reshape((-1)) * 2./hs - ws / hs
    y = 2 - (yv.reshape((-1)) * 2./hs) - 1
    z = depth.reshape((-1))

    nump = x.shape[0]

    m00 = projection[0,0]
    m11 = projection[1,1]

    xn = x*z / m00
    yn = y*z / m11
    zn = -z
    XY1 = np.concatenate([xn[:, None], yn[:, None], zn[:, None], np.ones((nump,1))], 1).transpose()
    # World coordinates
    XY = np.matmul(inv_view, XY1).transpose()

    x, y, z = XY[:, 0], XY[:, 1], XY[:, 2]
    if (len(img.shape) > 2):
        pos, col = np.array([x, y, z]).transpose(), np.array(img.reshape(-1, 3)) / 255
    else:
        pos, col = np.array([x, y, z]).transpose(), np.array(img.reshape(-1))
    image_location = np.array([ori_x.reshape(-1), ori_y.reshape(-1)]).transpose()
    if (len(col.shape) > 1):
        col = col[:, [2, 1, 0]]
    camera_pos = np.matmul(inv_view, np.array([0, 0, 0, 1]))
    if (type(mask) != type(None)):
        pos = pos[mask.reshape(-1)]
        col = col[mask.reshape(-1)]
        image_location = image_location[mask.reshape(-1)]
    if (far_away_remove == True):
        remain_id = [np.sqrt((pos[i][0] - camera_pos[0]) ** 2 + (pos[i][2] - camera_pos[2]) ** 2) <= remove_threshold for i in range(len(pos))]
        pos = pos[remain_id]
        col = col[remain_id]
        image_location = image_location[remain_id]
    return pos, col, image_location, camera_pos[:3]

def bbox_from_point(point):
    return [point for i in range(8)]

def relationship_detection(bbox_a, bbox_b, close_to_threshold = 2.5, outlog = False):
    r'''
        bbox_a: eight points of the first bbox, shape = (8, 3)
        bbox_b: eight points of the second bbox, shape = (8, 3)
        Assume the first bbox is the object, and the second bbox is the container
        the bbox's y-axis is parallel to the ground, (y is the second dimension of the point, means the height)
            i.e. (0, 1) = (1, 1) = (2, 1) = (3, 1) and (4, 1) = (5, 1) = (6, 1) = (7, 1)
        close_to_threshold: the distance threshold for CLOSE relationship, in virtual home doc is 1.5m, we set it to 1m because of inaccurte depth
        Return: the relationship between the two bboxes, an 3-dim array of bool:
            pos 0: whether ON
            pos 1: whether INSIDE
            pos 2: whether CLOSE

        Detecting INSIDE relationship: One way to detect whether the object is inside the container is to check whether all of the object's vertices are contained within the container. You can do this by checking whether each vertex of the object is within the convex hull of the container's vertices. If all vertices are contained within the convex hull, then the object is inside the container.

        Detecting ON relationship: If the object is not inside the container, it may still be on top of it. You can check if any of the vertices of the object are on the same plane as the top face of the container. To do this, you can check if the y-coordinate of the object's vertices is the same as the y-coordinate of the container's top face, and if the x- and z-coordinates of the object's vertices are within the range of the container's top face.

        Detecting CLOSE relationship: calculates the distance between the centers of the two bounding boxes and checks if it is below the threshold for closeness.
    '''
    if ("PointCloud" in str(type(bbox_a))): bbox_a = bbox_a.get_axis_aligned_bounding_box()
    if ("AxisAlignedBoundingBox" in str(type(bbox_a))): bbox_a = np.asarray(bbox_a.get_box_points())
    bbox_a = two_point_to_bbox(np.min(bbox_a, axis = 0), np.max(bbox_a, axis = 0))
    if ("PointCloud" in str(type(bbox_b))): 
        clip_pcd = bbox_b.crop(o3d.geometry.AxisAlignedBoundingBox([bbox_a[0][0] - 0.5, -10, bbox_a[0][2] - 0.5], [bbox_a[3][0] + 0.5, 10, bbox_a[3][2] + 0.5]))
        if (len(np.asarray(clip_pcd.points))) == 0: clip_pcd = bbox_b
        bbox_b = bbox_b.get_axis_aligned_bounding_box()
        clip_bbox_b = clip_pcd.get_axis_aligned_bounding_box()
    else:
        clip_bbox_b = bbox_b
    if ("AxisAlignedBoundingBox" in str(type(bbox_b))): bbox_b = np.asarray(bbox_b.get_box_points())
    if ("AxisAlignedBoundingBox" in str(type(clip_bbox_b))): clip_bbox_b = np.asarray(clip_bbox_b.get_box_points())
    bbox_b = two_point_to_bbox(np.min(bbox_b, axis = 0), np.max(bbox_b, axis = 0))
    clip_bbox_b = two_point_to_bbox(np.min(clip_bbox_b, axis = 0), np.max(clip_bbox_b, axis = 0))
    if (outlog): 
        print("bbox_a: ", bbox_a)
        print("bbox_b: ", bbox_b)
        print("clip_bbox_b: ", clip_bbox_b)
#    bbox_a = np.asarray(sorted(bbox_a, key=lambda x: x[1] * 1e6 + x[0] + x[2] * 1e-6))
#    bbox_b = np.asarray(sorted(bbox_b, key=lambda x: x[1] * 1e6 + x[0] + x[2] * 1e-6))

    assert(bbox_a.shape == (8, 3))
    assert(bbox_b.shape == (8, 3))

    down_rect_a = bbox_a[:4, [0,2]]
    up_rect_a = bbox_a[4:, [0,2]]
    assert(np.abs(down_rect_a - up_rect_a).max() < 0.25)
    rect_a = (down_rect_a + up_rect_a)/2
    down_y_a = bbox_a[:4, 1].mean()
    assert(np.abs(down_y_a - bbox_a[:4, 1]).max() < 0.25)
    up_y_a = bbox_a[4:, 1].mean()
    assert(np.abs(up_y_a - bbox_a[4:, 1]).max() < 0.25)
    down_rect_b = clip_bbox_b[:4, [0,2]]
    up_rect_b = clip_bbox_b[4:, [0,2]]
    assert(np.abs(down_rect_b - up_rect_b).max() < 0.25)
    rect_b = (down_rect_b + up_rect_b)/2
    down_y_b = clip_bbox_b[:4, 1].mean()
    assert(np.abs(down_y_b - clip_bbox_b[:4, 1]).max() < 0.25)
    up_y_b = clip_bbox_b[4:, 1].mean()
    assert(np.abs(up_y_b - clip_bbox_b[4:, 1]).max() < 0.25)
    def inside_rectangle(point, bbox2d, eps = 0):
        """
        Determines whether a point is inside a slanted rectangle in a 2D space.
        bbox2d: a 4x2 numpy array representing the four vertices of the rectangle.
        point: a 2-element numpy array representing the point to be checked.
        Returns True if point is inside the rectangle, False otherwise.
        """
        bbox2d = copy.deepcopy([bbox2d[i] for i in [0, 1, 3, 2]])
        bbox2d[0] += [-eps, -eps]
        bbox2d[1] += [-eps, eps]
        bbox2d[2] += [eps, eps]
        bbox2d[3] += [eps, -eps]
        AB = bbox2d[1] - bbox2d[0]
        BC = bbox2d[2] - bbox2d[1]
        CD = bbox2d[3] - bbox2d[2]
        DA = bbox2d[0] - bbox2d[3]
        AE = point - bbox2d[0]
        BE = point - bbox2d[1]
        CE = point - bbox2d[2]
        DE = point - bbox2d[3]
        if np.cross(AB, AE) <= 0 and np.cross(BC, BE) <= 0 and np.cross(CD, CE) <= 0 and np.cross(DA, DE) <= 0:
            return True
        else:
            return False
        
    def rect_inside_rect(bbox2din, bbox, eps = 0):
        return inside_rectangle(bbox2din[0], bbox, eps = eps) and inside_rectangle(bbox2din[1], bbox, eps = eps) and inside_rectangle(bbox2din[2], bbox, eps = eps) and inside_rectangle(bbox2din[3], bbox, eps = eps)
    is_ON = abs(down_y_a - up_y_b) < 0.6 and inside_rectangle(np.mean(rect_a, axis=0), rect_b, eps = 0.25) and (up_y_a + 0.1 > up_y_b)
    #update, to be checked
    is_INSIDE = down_y_a + 0.1 > down_y_b and up_y_a < up_y_b + 0.1 and rect_inside_rect(rect_a, rect_b, eps = 0.25)
    center_a = np.mean(bbox_a, axis=0)
    center_b = np.mean(bbox_b, axis=0)
    distance = np.linalg.norm(center_a[[0, 2]] - center_b[[0, 2]])
    is_CLOSE = distance < close_to_threshold
    
    return np.array([is_ON, is_INSIDE, is_CLOSE])

def two_point_to_bbox(a, b):
    return np.array([
        [a[0], a[1], a[2]],
        [a[0], a[1], b[2]],
        [b[0], a[1], a[2]],
        [b[0], a[1], b[2]],
        [a[0], b[1], a[2]],
        [a[0], b[1], b[2]],
        [b[0], b[1], a[2]],
        [b[0], b[1], b[2]]
    ])

def bbox_to_avg_3d_coords(self, ids, depth, camera_info, bbox):
    r'''
    depth: (H, W)
    camera_info: all hyperparameters of the camera, see image2coords for details,
    bbox: (x1, y1, x2, y2)
     
    return: 3d coordinates (x, y, z) (center or average is fine)
    '''
    #TODO Deprecated now
    img = np.zeros((depth.shape[0], depth.shape[1], 3))
    h,w, _ = depth.shape
    pos, col, _, _ = image2coords(img, depth, camera_info, far_away_remove = False)
    pos = pos.reshape(h,w,3)
    (x1, y1, x2, y2) = bbox

    x1, y1, x2, y2 = bbox
    bbox_ids = ids[x1:x2, y1:y2]  # get the ids within the bbox
    unique_ids, counts = np.unique(bbox_ids, return_counts=True)
    max_count_index = np.argmax(counts)
    max_id = unique_ids[max_count_index]
    if max_id == -1:
        raise ValueError("Max id of the bounding box is -1")
    mask = (ids == max_id)
    bbox_mask = mask[x1:x2, y1:y2]  # mask within bbox
    selected_pos = pos[x1:x2, y1:y2]  # 3D positions of pixels within bbox
    bbox_selected_pos = selected_pos[bbox_mask]  # 3D positions within bbox
    avg_pos = np.mean(bbox_selected_pos, axis=0)
    return avg_pos

def MCTS_to_language_convert(info, id_to_name_api = None):
    info = eval(info)
    if id_to_name_api is None: 
        name_padding = lambda x: ''
    else: name_padding = id_to_name_api
    language = {}
    if 'S' in info.keys(): # the format for mcts is put(In)_from_id_to_id
        act, from_id, to_id, goal_type = info['S'].split('_')
        from_id, to_id = int(from_id), int(to_id)
        if goal_type == '0':
            st = ' and I have not found it yet. '
        else:
            st = ' and I have found it. '
        if (act == 'put'):
            language['S'] = f'Now I want to put {name_padding(from_id)} <{from_id}> on {name_padding(to_id)} <{to_id}>, ' + st
        else:
            language['S'] = f'Now I want to put {name_padding(from_id)} <{from_id}> in {name_padding(to_id)} <{to_id}>, ' + st
    if 'E' in info.keys(): # the format for mcts is from_id on(in) to_id in room_id
        act, from_id, to_id, room_id = info['E']['relation_type'], info['E']['from_id'], info['E']['to_id'], info['E']['room_id']
        from_id, to_id = int(from_id), int(to_id)
        if (act == 'INSIDE'): act = 'in'
        elif (act == 'ON'): act = 'on'
        else: raise ValueError('Wrong action type')
        language['E'] = f'I successfully put {name_padding(from_id)} <{from_id}> {act} {name_padding(to_id)} <{to_id}>, and they are in {name_padding(room_id)} <{room_id}>. '
    if 'B' in info.keys():
        language['B'] = ''
        for data in info['B']:
            container, obj = data
            container, obj = int(container), [int(x) for x in obj]
            if len(obj) == 0:
                st = 'nothing'
            else:
                st = ''
                for i in range(len(obj)):
                    if i == len(obj) - 1:
                        st += f'{name_padding(obj[i])} <{obj[i]}>'
                    elif i == len(obj) - 2:
                        st += f'{name_padding(obj[i])} <{obj[i]}> and '
                    else:
                        st += f'{name_padding(obj[i])} <{obj[i]}>, '
            descript_container = f'{name_padding(container)} <{container}>. '
            if len(obj) > 1:
                st += f' are inside ' + descript_container
            else:
                st += f' is inside ' + descript_container
            language['B'] += st
    return f'{language}'

def language_to_MCTS_convert(language):
    msg = {}
    try:
        language = eval(language)
    except:
        return f'{msg}'
    if type(language) != dict: return f'{msg}'
    if 'S' in language.keys():
        data = language['S']
        number = re.findall(r'\d+', data)
        obj, container = number[0], number[1]
        if ' not found ' in data:
            goal_type = '0'
        else:
            goal_type = '1'
        if ' on ' in data:
            msg['S'] = f'put_{obj}_{container}_{goal_type}'
        else:
            msg['S'] = f'putIn_{obj}_{container}_{goal_type}'
    if 'E' in language.keys():
        data = language['E']
        number = re.findall(r'\d+', data)
        obj, container, room = int(number[0]), int(number[1]), int(number[2])
        if ' on ' in data:
            msg['E'] = {'relation_type': 'ON', 'from_id': obj, 'to_id': container, 'room_id': room}
        else: # be careful, this is not the same as 'in', 'in' occurs two times in the sentence
            msg['E'] = {'relation_type': 'INSIDE', 'from_id': obj, 'to_id': container, 'room_id': room}
    if 'B' in language.keys():
        data = language['B'].split('. ')[:-1]
        msg['B'] = []
        for d in data:
            number = re.findall(r'\d+', d)
            if 'nothing' in d: container, objs = int(number[-1]), []
            else: container, objs = int(number[-1]), [int(x) for x in number[:-1]]
            msg['B'].append((container, objs))
    return f'{msg}'

def vision_obs_to_graph(obs, memory):
    pass

if __name__ == '__main__':
    a1 = two_point_to_bbox([2, 9, 16], [4, 13, 20])
    b = two_point_to_bbox([3, 10, 17], [3.5, 12, 19])

    print(language_to_MCTS_convert("Alice, I found an apple in the kitchen. Could you check the cabinets in the kitchen and living room for pudding, juice, and cupcakes? Let\'s meet at the coffee table to put everything together."))