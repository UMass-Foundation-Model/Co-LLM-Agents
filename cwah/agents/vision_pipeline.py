import numpy as np
import open3d as o3d
import re
from . import utils
#from mmdet.apis import init_detector, inference_detector

'''

The detached pipeline for vision detection
Usage:
    1. create the agent_config
    2. create the vision_pipeline when resetting the agent, with the config and the init_obs as input
    3. for each step, call ''deal_with_obs(self, obs, last_action)'' to update the vision memory
    4. for each step, call ''get_graph()'' to get the symbolic graph of the current time step
'''

class agent_vision_config:
    def __init__(self, agent_type, char_index = 0, agent_id = 1, gt_seg = True, **kwargs):
        self.agent_type = agent_type
        self.char_index = char_index
        self.agent_id = agent_id
        self.gt_seg = gt_seg

class Vision_Pipeline:
    def __init__(self, config: agent_vision_config, init_obs):
        self.config = config
        self.object_info = {}
        self.object_relation = {}
        self.with_character_id = [config.agent_id]
        self.obs = init_obs
        self.last_action = None
        self.see_this_step = [] # The objects that are seen this step
        self.consider_upd = [] # The objects that are considered to be updated this step, the list contains the object id in last action
        for x in init_obs['room_info']:
            self.object_info[x['id']] = {
                'class_name': self.obs['get_class_name'](x['id']),
                'id': x['id'],
                'category': self.obs['get_category'](x['id']),
                'properties': self.obs['get_properties'](x['id']),
                'states': self.obs['get_states'](x['id'])
            }
        self.container_classes = [
            'bathroomcabinet',
            'kitchencabinet',
            'cabinet',
            'fridge',
            'stove',
            # 'kitchencounterdrawer',
			# 'coffeepot',
            'dishwasher',
            'microwave']
        config.gt_seg = True
        self.detector = None # deprecated
        if not config.gt_seg:
            config_file = 'detection_pipeline/mask_rcnn_r50-caffe_fpn_ms-1x_wah.py'
            checkpoint_file = 'detection_pipeline/epoch_12_4.21.pth'
            self.detector = init_detector(config_file, checkpoint_file, device='cuda:0')
            self.label_to_classname = ('plate', 'coffeetable', 'poundcake', 'wine', 'juice',
                         'pudding', 'apple', 'pancake', 'cutleryfork',
                         'kitchentable', 'cupcake', 'coffeepot', 'bathroomcabinet',
                         'kitchencabinet', 'cabinet', 'fridge', 'stove',
                         'dishwasher', 'microwave', 'character')
            print("init detector done")

        ''' For mask_rcnn: '''
    def find_relative_objects(self, image, frame_id = 0):
        return []
        # deprecated
        image = np.rot90(image, axes=(0, 1))
        # The image is rotated 90 degrees
        h, w = image.shape[:2]
        output = inference_detector(self.detector, image)
        labels = output.pred_instances.labels.to('cpu').numpy()
        masks = output.pred_instances.masks.to('cpu').numpy()
        scores = output.pred_instances.scores.to('cpu').numpy()
        bboxs = output.pred_instances.bboxes.to('cpu').numpy() # labels, [x, y]
        rot_bboxs = []
        for i in range(len(bboxs)):
            rot_bboxs.append([h - bboxs[i][3], bboxs[i][0], h - bboxs[i][1], bboxs[i][2]])
        image = np.rot90(image, axes=(1, 0))
        masks = np.rot90(masks, axes=(2, 1))
        object_info = []
        for i in range(len(scores)):
            if (scores[i] > 0.3):
                object_info.append({
                    'bbox': rot_bboxs[i],
                    'obj_name': self.label_to_classname[labels[i]],
                    'score': scores[i],
                    'frame_id': frame_id,
                    'mask': masks[i]
                })
        return object_info
    
    def obj_pcd_from_detection(self, obs, detect_info):
        r'''
            Update the point cloud of the object
            from rgbd image + bbox to point cloud
            idea: use the depth image to get the point cloud of the object region, than use clustering to get the point cloud of the object
            an object is reported as [<name> <position> <if container>]
            detect info:
                a list, and each element is {bbox: <bbox>, obj_name: <name>, score: <score>, id: <id>}
        '''
        pcds = []
        for i in range(len(detect_info)):
            pos, col, image_pos, _ = utils.image2coords(obs['bgr'][detect_info[i]['frame_id']], obs['depth'][detect_info[i]['frame_id']], obs['camera_info'][detect_info[i]['frame_id']], clip = detect_info[i]['bbox'], mask = detect_info[i]['mask'], far_away_remove = False)
            pcd = utils.read_pcd_from_point_array(pos, col)
            pcd = pcd.voxel_down_sample(voxel_size=0.01)
            pos = np.asarray(pcd.points)
            col = np.asarray(pcd.colors)
            labels = np.array(pcd.cluster_dbscan(eps=0.025, min_points=1))
            vis_labels = labels[labels >= 0]
            if (len(vis_labels) == 0): # The object is too far, leave it later.
                continue
            true_label = np.argmax(np.bincount(vis_labels))
            labels = (labels == true_label) # The main cluster
            pos = pos[labels]
            col = col[labels]
            pcd = {'pos': pos, 'col': col, 'name': detect_info[i]['obj_name'], 'score': detect_info[i]['score'], 'id': detect_info[i]['id']}
            pcds.append(pcd)
            self.see_this_step.append(pcd['id'])
        return pcds

    def update_object_info(self, pcd):
        if pcd['id'] not in self.object_info.keys():
            self.object_info[pcd['id']] = {
                'pcd': utils.read_pcd_from_point_array(pcd['pos'], pcd['col']),
                'class_name': pcd['name'],
                'properties': self.obs['get_properties'](pcd['id']),
                'category': self.obs['get_category'](pcd['id']),
                'states': self.obs['get_states'](pcd['id'])
            }
        else:
            if pcd['id'] in self.with_character_id or 'pcd' not in self.object_info[pcd['id']]: return
            self.object_info[pcd['id']]['pcd'] = utils.read_pcd_from_point_array(pcd['pos'], pcd['col'])
            self.object_info[pcd['id']]['pcd'] = self.object_info[pcd['id']]['pcd'].voxel_down_sample(voxel_size=0.01)
    
    def obj_pcd_from_seg(self, obs):
        seg_map = {}
        for i in range(len(obs['seg_info'])):
            pos, col, image_location, camera_pos = utils.image2coords(obs['seg_info'][i], obs['depth'][i], obs['camera_info'][i], far_away_remove = False)
            for j in range(len(col)):
                if (col[j] >= 0):
                    if (col[j] in seg_map.keys()):
                        seg_map[col[j]]['pos'].append(pos[j])
                    else:
                        seg_map[col[j]] = {'id': int(col[j]), 'pos': [pos[j]], 'name': self.obs['get_class_name'](col[j]), 'col': None}
        removed = []
        len_removed = {}
        for id in seg_map.keys():
            pcd = utils.read_pcd_from_point_array(seg_map[id]['pos'])
            pcd = pcd.voxel_down_sample(voxel_size=0.01)
            pos = np.asarray(pcd.points)
            labels = np.array(pcd.cluster_dbscan(eps=0.1, min_points=3))
            vis_labels = labels[labels >= 0]
            if (len(vis_labels) == 0 or len(labels) < 5):
                # The object is too far, leave it later.
                # depends on image size, leave it later.
                removed.append(id)
                len_removed[id] = len(labels)
                continue
            else:
                self.see_this_step.append(id)
            true_label = np.argmax(np.bincount(vis_labels))
            labels = (labels == true_label) # the main cluster
            pos = pos[labels]
            seg_map[id]['pos'] = pos
        for id in removed:
            seg_map.pop(id)
        return seg_map
    
    def get_graph(self):
        r'''
            get the graph from the object info
        '''
        # only see the objects can we update the object info
        self.see_this_step += self.with_character_id
        self.see_this_step.append(self.obs['current_room'])
        for id in self.see_this_step:
            self.object_info[id]['properties'] = self.obs['get_properties'](id)
            self.object_info[id]['category'] = self.obs['get_category'](id)
            self.object_info[id]['states'] = self.obs['get_states'](id)
        for id in self.consider_upd:
            assert id in self.object_info.keys()
            self.object_info[id]['properties'] = self.obs['get_properties'](id)
            self.object_info[id]['category'] = self.obs['get_category'](id)
            self.object_info[id]['states'] = self.obs['get_states'](id)
        self.consider_upd = []

        all_nodes = [{'id': x[0], **x[1]} for x in self.object_info.items()]
        nodes = self.obs['nodes_in_same_room'](all_nodes)
        self.visiable_ids = [x['id'] for x in nodes]

        detected_edges = []
        inside_node = np.zeros(len(nodes))

        close_object = []
        # detect the edges between the agent and the object
        for i in range(len(nodes)):
            if nodes[i]['category'] in ['Rooms'] or nodes[i]['class_name'] in ['character']: continue
            self.object_relation[(self.config.agent_id, nodes[i]['id'])] = [0, 0, 0]
            self.object_relation[(nodes[i]['id'], self.config.agent_id)] = [0, 0, 0]
            bbox_my = utils.bbox_from_point(self.obs['location'])
            if nodes[i]['id'] not in self.see_this_step and nodes[i]['id'] not in self.with_character_id: continue
            if nodes[i]['id'] in self.with_character_id: bbox_i = utils.bbox_from_point(self.obs['location'])
            else: bbox_i = nodes[i]['pcd'].get_axis_aligned_bounding_box()
            self.object_relation[(self.config.agent_id, nodes[i]['id'])] = [0, 0, utils.relationship_detection(bbox_my, bbox_i)[2]]
            self.object_relation[(nodes[i]['id'], self.config.agent_id)] = [0, 0, utils.relationship_detection(bbox_i, bbox_my)[2]]
            if self.object_relation[(self.config.agent_id, nodes[i]['id'])][2]:
                close_object.append(nodes[i]['id'])
        # If an object is close in vision, we calculate the relationship between the ojbect.
        # The truly close relationship come from gt.

        # detect the edges between objects
        for i in range(len(nodes)):
            if nodes[i]['id'] not in close_object: continue
            # not detect the edges of far-away objects
            for j in range(len(nodes)):
                if i == j: continue
                if nodes[i]['category'] in ['Rooms'] or nodes[j]['category'] in ['Rooms']: continue
                if nodes[i]['id'] not in self.see_this_step and nodes[i]['id'] not in self.with_character_id: continue
                if nodes[j]['id'] not in self.see_this_step and nodes[j]['id'] not in self.with_character_id: continue
                if nodes[i]['id'] in self.with_character_id: bbox_i = utils.bbox_from_point(self.obs['location'])
                elif 'pcd' in nodes[i]: bbox_i = nodes[i]['pcd']
                else: continue
                if nodes[j]['id'] in self.with_character_id: bbox_j = utils.bbox_from_point(self.obs['location'])
                elif 'pcd' in nodes[j]: bbox_j = nodes[j]['pcd']
                else: continue
                temp_relation = utils.relationship_detection(bbox_i, bbox_j)
                if 'CAN_OPEN' not in nodes[j]['properties'] or 'CAN_OPEN' in nodes[i]['properties'] or nodes[j]['class_name'] not in self.container_classes: temp_relation[1] = 0
                if 'SURFACES' not in nodes[j]['properties']: temp_relation[0] = 0
                if temp_relation[2] == 1 and (nodes[i]['id'], nodes[j]['id']) in self.object_relation.keys():
                    self.object_relation[(nodes[i]['id'], nodes[j]['id'])] = np.array([temp_relation[x] or self.object_relation[(nodes[i]['id'], nodes[j]['id'])][x] for x in range(3)])
                else:
                    self.object_relation[(nodes[i]['id'], nodes[j]['id'])] = temp_relation
                #If still close and not grab, the error is due to detection error.
                if nodes[i]['id'] in self.with_character_id: self.object_relation[(nodes[i]['id'], nodes[j]['id'])][0: 2] = 0
                #If (nodes[i]['id'] in [386, 390, 391, 384] and nodes[j]['id'] in [130]):
                #    print(nodes[i]['id'], nodes[j]['id'], self.object_relation[(nodes[i]['id'], nodes[j]['id'])])
                #    print(utils.relationship_detection(bbox_i, bbox_j, outlog = True))

        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j: continue
                if nodes[i]['category'] in ['Rooms'] or nodes[j]['category'] in ['Rooms']: continue
                if (nodes[i]['id'], nodes[j]['id']) not in self.object_relation.keys(): continue
                relation = self.object_relation[(nodes[i]['id'], nodes[j]['id'])]
                if relation[0] and nodes[i]['category'] not in ['Rooms', 'Characters']:
                    detected_edges.append({'from_id': nodes[i]['id'], 'to_id': nodes[j]['id'], 'relation_type': 'ON'})
                if relation[1] and nodes[i]['category'] not in ['Rooms', 'Characters'] and inside_node[i] == 0:
                    detected_edges.append({'from_id': nodes[i]['id'], 'to_id': nodes[j]['id'], 'relation_type': 'INSIDE'})
                    inside_node[i] = nodes[j]['id']
        
        for i in range(len(nodes)):
            if inside_node[i] == 0 and nodes[i]['category'] not in ['Rooms']:
                detected_edges.append({'from_id': nodes[i]['id'], 'to_id': self.obs['current_room'], 'relation_type': 'INSIDE'})

        for node in self.obs['nodes']: 
            if node['id'] not in [x['id'] for x in nodes]: nodes.append(node)
        all_nodes = [x['id'] for x in nodes]
        for edge in self.obs['edges']:
            if edge['from_id'] not in all_nodes: continue
            if edge['to_id'] not in all_nodes and self.obs['get_category'](edge['to_id']) != 'Rooms': continue
            if edge['from_id'] not in self.visiable_ids or edge['relation_type'] not in ['INSIDE']: detected_edges.append(edge)
        # Only consider the visible special edges, such as hold, CLOSE

        for node in nodes:
            if 'pcd' in node.keys(): node.pop('pcd')

        detected_edges = [{**x, 'obs': True} for x in detected_edges]
        return self.obs['remove_duplicate_graph']({'nodes': nodes, 'edges': detected_edges})
    
    def deal_with_obs(self, obs, last_action = None):
        self.obs = obs
        self.last_action = last_action
        self.see_this_step = []
        if 'MCTS' in self.config.agent_type:
            if obs['messages'] is not None and len(obs['messages']) > 1 and obs['messages'][1 - self.config.char_index]:
                msg = eval(utils.language_to_MCTS_convert(obs['messages'][1 - self.config.char_index]))
                if 'E' in msg.keys():
                        def add_distinct(l1, l2):
                            for i in l2:
                                for j in l1:
                                    if i == j:
                                        break
                                else: l1.append(i)
                        add_relation = {}
                        if msg['E']['relation_type'] == 'INSIDE':
                            add_relation['edges'] = [{'from_id': msg['E']['from_id'], 'to_id': msg['E']['to_id'], 'relation_type': 'INSIDE'},
                                                     {'from_id': msg['E']['to_id'], 'to_id': msg['E']['room_id'], 'relation_type': 'INSIDE'}]
                        elif msg['E']['relation_type'] == 'ON':
                            add_relation['edges'] = [{'from_id': msg['E']['from_id'], 'to_id': msg['E']['to_id'], 'relation_type': 'ON'},
                                                     {'from_id': msg['E']['to_id'], 'to_id': msg['E']['room_id'], 'relation_type': 'INSIDE'},
                                                     {'from_id': msg['E']['from_id'], 'to_id': msg['E']['room_id'], 'relation_type': 'INSIDE'}]
                        relative_id = list(set([x['from_id'] for x in add_relation['edges']] + [x['to_id'] for x in add_relation['edges']]))
                        add_relation['nodes'] = [
                            {'class_name': self.obs['get_class_name'](x),
                            'id': x,
                            'category': self.obs['get_category'](x),
                            'properties': self.obs['get_properties'](x),
                            'states': self.obs['get_states'](x)}
                            for x in relative_id]
                        add_distinct(obs['nodes'], add_relation['nodes'])
                        add_distinct(obs['edges'], add_relation['edges'])
                        no_change_id = [x['to_id'] for x in add_relation['edges']]
                        ids = [x['id'] for x in add_relation['nodes'] if x['id'] not in no_change_id]
                        id2node_sent = {x['id']: x for x in add_relation['nodes']}
                        removed = []
                        removed_edges = []
                        for item_id in self.object_info.keys():
                            if (item_id in ids):
                                removed.append(item_id)
                        for item_id in removed:
                            self.object_info.pop(item_id)
                        for item_id in ids:
                            self.object_info[item_id] = {
                                'pcd': o3d.geometry.PointCloud(),
                                'class_name': id2node_sent[item_id]['class_name'],
                                'properties': self.obs['get_properties'](item_id),
                                'category': self.obs['get_category'](item_id),
                                'states': self.obs['get_states'](item_id)
                            }
                        for edge in self.object_relation.keys():
                            if (edge[0] in removed or edge[1] in removed):
                                removed_edges.append(edge)
                        for edge in removed_edges:
                            self.object_relation.pop(edge)
                        for edge in add_relation['edges']:
                            if (edge['relation_type'] == 'ON'): relation = [True, False, True]
                            elif (edge['relation_type'] == 'INSIDE'): relation = [False, True, True]
                            else: relation = [False, False, True]
                            self.object_relation[(edge['from_id'], edge['to_id'])] = relation

        self.consider_upd = []
        if self.last_action is not None and 'message' not in self.last_action and 'Turn' not in self.last_action:
            if 'grab' in self.last_action:
                grabbed_objects_id = [e['to_id'] for e in self.obs['edges'] if e['from_id'] == self.config.agent_id and e['relation_type'] in ['HOLDS_RH', 'HOLDS_LH']]
                to_grab_id = int(self.last_action.split(' ')[2][1: -1])
                if to_grab_id not in grabbed_objects_id:
                    print(f"Failed to grab {to_grab_id}")
                else:
                    self.with_character_id.append(to_grab_id)
                    self.consider_upd.append(to_grab_id)
                    if to_grab_id in self.object_info.keys():
                        if 'pcd' in self.object_info[to_grab_id]:
                            self.object_info[to_grab_id]['pcd'] = o3d.geometry.PointCloud()
                    else:
                        print('grab an object that is not in the object_info')
                    # Clear the pcd
            if 'put' in self.last_action:
                self.with_character_id.remove(int(self.last_action.split(' ')[2][1: -1]))
                self.consider_upd.append(int(self.last_action.split(' ')[2][1: -1]))
            if 'putback' in self.last_action:
                self.object_relation[(int(self.last_action.split(' ')[2][1: -1]), int(self.last_action.split(' ')[4][1: -1]))] = [1, 0, 1]
                self.consider_upd.append(int(self.last_action.split(' ')[2][1: -1]))
            if 'putin' in self.last_action:
                self.object_relation[(int(self.last_action.split(' ')[2][1: -1]), int(self.last_action.split(' ')[4][1: -1]))] = [0, 1, 1]
                self.consider_upd.append(int(self.last_action.split(' ')[2][1: -1]))
            if 'open' in self.last_action:
                self.consider_upd.append(int(self.last_action.split(' ')[2][1: -1]))
        
        self.obs = obs
        self.obs['location'][1] = 1
        # Update object info
        if 'seg_info' not in obs.keys():
            object_info, ground_info = [], []
            for i in range(len(obs['bgr'])): object_info += self.find_relative_objects(obs['bgr'][i], frame_id = i)
            for i in range(len(object_info)):
                ground_id, ground_name = obs['bbox_2d_to_id'](object_info[i]['bbox'], frame_id = object_info[i]['frame_id'], mask = object_info[i]['mask'])
                if (ground_id != -1):
                    object_info[i]['obj_name'] = ground_name
                    ground_info.append({**object_info[i], 'id': ground_id})
            pcds = self.obj_pcd_from_detection(obs, ground_info)
            for pcd in pcds: self.update_object_info(pcd)
        else:
            pcds = self.obj_pcd_from_seg(obs)
            for x in pcds.keys(): self.update_object_info(pcds[x])