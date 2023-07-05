import json
from scipy.spatial.transform import Rotation as R

import pickle as pkl
import ipdb
import os

import plotly.graph_objects as go
import plotly.io
import pdb
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle
import plotly.offline
from tqdm import tqdm
import cv2
import glob
import subprocess
import re
import numpy as np

dict_info = {
    "objects_inside": [
        "toilet", "bathroom_cabinet", "kitchencabinets",
        "bathroom_counter", "kitchencounterdrawer", "cabinet", "fridge", "oven", "dishwasher", "microwave"],

    "objects_surface": ["bathroomcabinet",
                        "bathroomcounter",
                        "bed",
                        "bench",
                        "boardgame",
                        "bookshelf",
                        "cabinet",
                        "chair",
                        "coffeetable",
                        "cuttingboard",
                        "desk",
                        "fryingpan",
                        "kitchencabinets",
                        "kitchencounter",
                        "kitchentable",
                        "mousemat",
                        "nightstand",
                        "oventray",
                        "plate",
                        "radio",
                        "sofa",
                        "stove",
                        "towelrack"],
    "objects_grab": [
        "pudding", "juice", "pancake", "apple",
        "book", "coffeepot", "cupcake", "cutleryfork", "dishbowl", "milk",
        "milkshake", "plate", "poundcake", "remotecontrol", "waterglass", "wine", "wineglass"
    ]
}
#color_code = map()

#def create_sphere(nodes, color="yellow", opacity=0.8):
#    c, b = node['bounding_box']['center'], node['bounding_box']['size']
##    if cont:
#    circle_data = go.
#    return circle_data

def create_cube(nodes, color='lightpink', opacity=0.1, cont=False):
    c, b = nodes['bounding_box']['center'], nodes['bounding_box']['size']

    if cont:
        xp = [c[0] - b[0] / 2., c[0] + b[0] / 2.] * 4
        zp = [c[1] - b[1] / 2.] * 4 + [c[1] + b[1] / 2.] * 4
        yp = [c[2] - b[2] / 2.] * 2 + [c[2] + b[2] / 2.] * 2 + [c[2] - b[2] / 2.] * 2 + [c[2] + b[2] / 2.] * 2
        indices = [0, 1, 3, 2, 0, 4, 5, 1, 5, 7, 3, 7, 6, 2, 6, 4]
        x = [xp[it] for it in indices]
        y = [yp[it] for it in indices]
        z = [zp[it] for it in indices]
        cube_data = go.Scatter3d(x=x, y=y, z=z, showlegend=False, opacity=opacity, mode='lines', hoverinfo='skip',
                                 marker={'color': color})
    else:
        x = [c[0] - b[0] / 2., c[0] + b[0] / 2.] * 4
        z = [c[1] - b[1] / 2.] * 4 + [c[1] + b[1] / 2.] * 4
        y = [c[2] - b[2] / 2.] * 2 + [c[2] + b[2] / 2.] * 2 + [c[2] - b[2] / 2.] * 2 + [c[2] + b[2] / 2.] * 2
        i = [0, 3, 4, 7, 0, 6, 1, 7, 0, 5, 2, 7]
        j = [1, 2, 5, 6, 2, 4, 3, 5, 4, 1, 6, 3]
        k = [3, 0, 7, 4, 6, 0, 7, 1, 5, 0, 7, 2]
        cube_data = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=opacity)
    return cube_data


def create_points(nodes, color='red'):
    centers = [n['bounding_box']['center'] for n in nodes]
    class_and_ids = ['{}.{}'.format(node['class_name'], node['id']) for node in nodes]
    x, z, y = zip(*centers)
    scatter_data = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker={'size': 3, 'color': color},
                                showlegend=False, hovertext=class_and_ids)
    return scatter_data


def plot_graph(graph, char_id=1, visible_ids=None, action_ids=None):
    nodes_interest = [node for node in graph['nodes'] if 'GRABBABLE' in node['properties']]
    container_surf = dict_info['objects_inside'] + dict_info['objects_surface']
    container_and_surface = [node for node in graph['nodes'] if node['class_name'] in container_surf]
    grabbed_obj = [node for node in graph['nodes'] if node['class_name'] in dict_info['objects_grab']]
    rooms = [node for node in graph['nodes'] if 'Rooms' == node['category']]

    # Character
    char_node = [node for node in graph['nodes'] if node['id'] == char_id][0]

    room_data = [create_cube(n, color='lightpink', cont=True, opacity=0.2) for n in rooms]

    # containers and surfaces
    visible_nodes = [node for node in graph['nodes'] if node['id'] in visible_ids]
    action_nodes = [node for node in graph['nodes'] if node['id'] in action_ids]

    goal_nodes = [node for node in graph['nodes'] if node['class_name'] == 'cupcake']
    #object_data2 = [create_cube(n, color='green', cont=True, opacity=0.2) for n in grabbed_obj]
    object_data = [create_cube(n, color='blue', cont=True, opacity=0.1) for n in container_and_surface]
    object_data_vis = [create_cube(n, color='green', cont=True, opacity=0.2) for n in visible_nodes]
    object_data_action = create_points(action_nodes, color='pink')

    fig = go.Figure()

    fig.add_traces(data=create_cube(char_node, color="yellow", opacity=0.8))
    fig.add_traces(data=object_data)
    fig.add_traces(data=object_data_vis)
    fig.add_traces(data=object_data_action)
    fig.add_traces(data=room_data)
    fig.add_traces(data=create_points(goal_nodes))

    fig.update_layout(scene_aspectmode='data')
    return fig


def save_graph(graph, char_id=1, visible_ids=None, action_ids=None):
    fig = plot_graph(graph, char_id, visible_ids, action_ids)
    html_str = plotly.io.to_html(fig, include_plotlyjs=False, full_html=False)
    return html_str

#
# if __name__ == '__main__':
#     import json
#
#     with open('graph_seq2.json', 'r') as f:
#         graphs = json.load(f)
#     return_str = save_graph(graphs[0])
#     pdb.set_trace()



##### 2D PLOTTING ####

def get_bounds(bounds):
    minx, maxx = None, None
    miny, maxy = None, None
    for bound in bounds:
        bgx, sx = bound['center'][0] + bound['size'][0] / 2., bound['center'][0] - bound['size'][0] / 2.
        bgy, sy = bound['center'][2] + bound['size'][2] / 2., bound['center'][2] - bound['size'][2] / 2.
        minx = sx if minx is None else min(minx, sx)
        miny = sy if miny is None else min(miny, sy)
        maxx = bgx if maxx is None else max(maxx, bgx)
        maxy = bgy if maxy is None else max(maxy, bgy)
    return (minx, maxx), (miny, maxy)


def add_box(nodes, args_shape):
    rectangles = []
    centers = [[], []]
    for node in nodes:
        cx, cy = node['bounding_box']['center'][0], node['bounding_box']['center'][2]
        w, h = node['bounding_box']['size'][0], node['bounding_box']['size'][2]
        minx, miny = cx - w / 2., cy - h / 2.
        centers[0].append(cx)
        centers[1].append(cy)
        if args_shape is not None:
            rectangles.append(
                Rectangle((minx, miny), w, h, **args_shape)
            )
    return rectangles, centers

def add_circle(nodes, args_shape):
    circles = []
    centers = [[], []]
    for node in nodes:
        cx, cy = node['bounding_box']['center'][0], node['bounding_box']['center'][2]
        if 'radius' not in args_shape: 
            w, h = node['bounding_box']['size'][0], node['bounding_box']['size'][2]
            radius = max(h,w)/2.
            args_shape['radius'] = radius

        centers[0].append(cx)
        centers[1].append(cy)
        if args_shape is not None:
            circles.append(
                    Circle((cx,cy), **args_shape)
                              )
    return circles, centers

def add_boxes(nodes, ax, points=None, args_shape=None):
    rectangles = []
    rectangles_class, center = add_box(nodes, args_shape)
    rectangles += rectangles_class
    if points is not None:
        ax.scatter(center[0], center[1], **points)
    if args_shape is not None and len(rectangles) > 0:
        ax.add_patch(rectangles[0])
        collection = PatchCollection(rectangles, match_original=True)
        ax.add_collection(collection)

def add_circles(nodes, ax, points=None, args_shape=None):
    circles = []
    circles_class, center = add_circle(nodes, args_shape)
    circles += circles_class
    if points is not None:
        ax.scatter(center[0], center[1], **points)
    if args_shape is not None and len(circles) > 0:
        ax.add_patch(circles[0])
        collection = PatchCollection(circles, match_original=True)
        ax.add_collection(collection)


prev_cx, prev_cy = {},{}
def add_arrow(nodes, args_shape):
    arrows = []
    centers = [[], []]
    if 'node_radius' not in args_shape:
        node_radius = None
    else:
        node_radius = args_shape['node_radius']
    del args_shape['node_radius'] 
    for node in nodes:
        if not node['id'] not in prev_cx and 'obj_transform' not in node.keys():
            
            cx, cy = node['bounding_box']['center'][0], node['bounding_box']['center'][2]
            w, h = node['bounding_box']['size'][0], node['bounding_box']['size'][2]
            prev_cx[node['id']], prev_cy[node['id']] = cx, cy

        else:
            curr_cx, curr_cy = node['bounding_box']['center'][0], node['bounding_box']['center'][2]
            if 'obj_transform' in node:
                rot = node['obj_transform']['rotation']
                # print('rotation: ', rot)
                rot = R.from_quat(rot)
                euler = rot.as_euler('xzy')
                # dchange = np.sin(euler[1])*np.cos(euler[0]), np.cos(euler[1])*np.sin(euler[0])
                # dchange = np.sin(euler[1]+euler[0]), np.cos(euler[1]+euler[0])
                x = np.cos(euler[2])*np.cos(euler[1])
                y = np.sin(euler[2])*np.cos(euler[1])
                z = np.sin(euler[1])
                dchange = y, x
                # print('euler: ', euler)
                # print('dchange 1: x: {}, y: {}'.format(dchange[0], dchange[1]))
            else:
                dchange = [curr_cx - prev_cx[node['id']], curr_cy - prev_cy[node['id']]]
                dchange = dchange / np.sqrt(np.sum(np.square(dchange)))
            
            if 'node_radius' is None:
                w, h = node['bounding_box']['size'][0], node['bounding_box']['size'][2]
                node_radius = max(h,w)/2.
            
            prev_cx[node['id']], prev_cy[node['id']] = curr_cx, curr_cy
            
            if 'radius' not in args_shape:
                radius = 0.2 * node_radius
                args_shape['radius'] = radius

            dist_c = 0.8 * node_radius

            cx, cy = curr_cx + dist_c * dchange[0], curr_cy + dist_c * dchange[1]
            centers[0].append(cx)
            centers[1].append(cy)

            if args_shape is not None:
                arrows.append(
                        Circle((cx,cy), **args_shape)
                              )
        return arrows, centers





def add_arrows(nodes, ax, points=None, args_shape=None):
    arrows = []
    arrows_class, center = add_arrow(nodes, args_shape)
    arrows += arrows_class
    if points is not None:
        ax.scatter(center[0], center[1], **points)
    if args_shape is not None and len(arrows) > 0:
        ax.add_patch(arrows[0])
        collection = PatchCollection(arrows, match_original=True)
        ax.add_collection(collection)


def plot_graph_2d_v2(graph, char_id, visible_ids, action_ids, goal_ids, display_furniture=True):

    
    id2node = {node['id']: node for node in graph['nodes']}
    goals = [node for node in graph['nodes'] if node['id'] in goal_ids]
    container_surf = dict_info['objects_inside'] + dict_info['objects_surface']
    container_and_surface = [node for node in graph['nodes'] if node['class_name'] in container_surf]

    #grabbed_obj = [node for node in graph['nodes'] if node['class_name'] in dict_info['objects_grab']]
    rooms = [node for node in graph['nodes'] if 'Rooms' == node['category']]


    # containers and surfaces
    visible_nodes = [node for node in graph['nodes'] if node['id'] in visible_ids and node['category'] != 'Rooms']
    action_nodes = [node for node in graph['nodes'] if node['id'] in action_ids and node['category'] != 'Rooms']
    
    visible_nodes_object = [node for node in visible_nodes if node['class_name'] not in container_surf]
    visible_nodes_container = [node for node in visible_nodes if node['class_name'] in container_surf]

    # Character
    char_nodes = [id2node[ch_id] for ch_id in char_id]
    plt.close('all');
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    add_boxes(rooms, ax, points=None, args_shape={'alpha': 0.1})

    if display_furniture:
        add_boxes(container_and_surface, ax, points=None, args_shape={'fill': False,
                                                                            'edgecolor': 'blue', 'alpha': 0.3})
#    add_boxes([char_node], ax, points=None, rect={'facecolor': 'yellow', 'edgecolor': 'yellow', 'alpha': 0.7})
    
    colors = ['blue', 'magenta']
    for char_it, char_node in enumerate(char_nodes):
        add_circles([char_node], ax, points=None, args_shape={'facecolor': colors[char_it], 'edgecolor': colors[char_it], 'alpha': 0.7, 'radius': 0.5})
        add_arrows([char_node], ax, points=None, args_shape={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 1.0, 'radius': 0.1, 'node_radius': 0.5})
    
    visible_nodes_nochar = [node for node in visible_nodes_object if node['class_name'] != 'character']
    for node in visible_nodes_nochar:
        if node['bounding_box'] == None:
            print(node)

    if len(visible_nodes_nochar) > 0:
        add_boxes(visible_nodes_nochar, ax, args_shape={'fill': False,
                                                        'edgecolor': 'blue', 'alpha': 1.0})
    if len(visible_nodes_container) > 0:
        add_boxes(visible_nodes_container, ax, points=None, args_shape={'fill': False,
            'edgecolor': 'blue', 'alpha': 0.6})
    
    add_boxes(goals, ax, args_shape={'fill': False, 'edgecolor': 'orange', 'alpha': 1.0, 'linewidth': 2.0})
    add_boxes(action_nodes, ax, points=None, args_shape={'fill': False, 'edgecolor': 'red', 'alpha': 1.0, 'linewidth': 2.7})


    bad_classes = ['character']

    ax.set_aspect('equal')
    bx, by = get_bounds([room['bounding_box'] for room in rooms])

    maxsize = max(bx[1] - bx[0], by[1] - by[0])
    gapx = (maxsize - (bx[1] - bx[0])) / 2.
    gapy = (maxsize - (by[1] - by[0])) / 2.

    ax.set_xlim(bx[0]-gapx, bx[1]+gapx)
    ax.set_ylim(by[0]-gapy, by[1]+gapy)
    ax.apply_aspect()
    return fig

def plot_graph_2d(graph, char_id, visible_ids, action_ids, goal_ids):


    #nodes_interest = [node for node in graph['nodes'] if 'GRABBABLE' in node['properties']]
    goals = [node for node in graph['nodes'] if node['id'] in goal_ids]
    container_surf = dict_info['objects_inside'] + dict_info['objects_surface']
    container_and_surface = [node for node in graph['nodes'] if node['class_name'] in container_surf]

    #grabbed_obj = [node for node in graph['nodes'] if node['class_name'] in dict_info['objects_grab']]
    rooms = [node for node in graph['nodes'] if 'Rooms' == node['category']]


    # containers and surfaces
    visible_nodes = [node for node in graph['nodes'] if node['id'] in visible_ids and node['category'] != 'Rooms']
    action_nodes = [node for node in graph['nodes'] if node['id'] in action_ids and node['category'] != 'Rooms']

    goal_nodes = [node for node in graph['nodes'] if node['class_name'] == 'cupcake']

    # Character
    char_node = [node for node in graph['nodes'] if node['id'] == char_id][0]

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    add_boxes(rooms, ax, points=None, args_shape={'alpha': 0.1})
    add_boxes(container_and_surface, ax, points=None, args_shape={'fill': False,
                                                                        'edgecolor': 'blue', 'alpha': 0.3})
#    add_boxes([char_node], ax, points=None, rect={'facecolor': 'yellow', 'edgecolor': 'yellow', 'alpha': 0.7})

    add_circles([char_node], ax, points=None, args_shape={'facecolor': 'yellow', 'edgecolor': 'yellow', 'alpha': 0.7})
    add_arrows([char_node], ax, points=None, args_shape={'facecolor': 'black', 'edgecolor': 'black', 'alpha': 0.7})
    
    add_boxes(visible_nodes, ax, points={'s': 2.0, 'alpha': 1.0}, args_shape={'fill': False,
                                                                         'edgecolor': 'green', 'alpha': 1.0})
    add_boxes(goals, ax, points={'s':  100.0, 'alpha': 1.0, 'edgecolors': 'orange', 'facecolors': 'none', 'linewidth': 3.0})
    add_boxes(action_nodes, ax, points={'s': 3.0, 'alpha': 1.0, 'c': 'red'})


    bad_classes = ['character']

    ax.set_aspect('equal')
    bx, by = get_bounds([room['bounding_box'] for room in rooms])

    maxsize = max(bx[1] - bx[0], by[1] - by[0])
    gapx = (maxsize - (bx[1] - bx[0])) / 2.
    gapy = (maxsize - (by[1] - by[0])) / 2.

    ax.set_xlim(bx[0]-gapx, bx[1]+gapx)
    ax.set_ylim(by[0]-gapy, by[1]+gapy)
    ax.apply_aspect()
    return fig

def save_graph_2d(img_name, graph, visible_ids, action_ids, goal_ids, char_id=1):
    fig = plot_graph_2d(graph['graph'], char_id, visible_ids, action_ids, goal_ids)
    # plt.axis('scaled')
    # print([node['bounding_box']['size'] for node in graph['nodes'] if node['id'] == 1][0])
    fig.tight_layout()

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (30, 30)
    fontScale              = 0.5
    fontColor              = (0, 0, 0)
    lineType               = 2

    fig.savefig(img_name)
    curr_im = cv2.imread(img_name)
    cv2.putText(
        curr_im, graph['instruction'][0],
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
    cv2.imwrite(img_name, curr_im)
    plt.close(fig)


####################


def render(el):
    if el is None:
        return "None"
    if type(el) == list:
        ncontent = [x.replace('<', '&lt').replace('>', '&gt').replace('[', '&lbrack;').replace(']', '&rbrack;') for x in el]
        return ''.join(['<span style="display:inline-block; width: 150px">'+x+'</span>' for x in ncontent])
    if type(el) == str:
        el_html = el.replace('<', '&lt').replace('>', '&gt').replace('[', '&lbrack;').replace(']', '&rbrack;')
        return el_html
    if type(el) == dict:
        visible_ids = el['visible_ids'][0]
        action_ids = [t for t in el['action_ids'][0] if t in visible_ids]

        return save_graph(el['graph'][0], visible_ids=visible_ids, action_ids=action_ids)
    else:
        return el.render()

class html_img:
    def __init__(self, src):
        self.src = src

    def render(self):
        return '<img src="{}" style="height: 600px">'.format(self.src)

def html_table(titles, max_rows, column_info, column_style=None):
    header = ''.join(['<th>{}</th>'.format(title) for title in titles])
    table_header = '<tr> {} </tr>'.format(header)

    table_contents = ''
    widths = column_style

    for row_id in range(max_rows):
        table_contents += '<tr>'
        for it in range(len(column_info)):
            if widths is not None:
                w = widths[it]
            else:
                w = ''
            if len(column_info[it]) > row_id:
                el = column_info[it][row_id]
                table_contents += '<td style="{}"> {} </td>'.format(w, render(el))
            else:
                table_contents += '<td style="{}"></td>'.format(w)
        table_contents += '</tr>'
    table_ep = '<table style="border-width: 2px; color: black; border-style: solid" > {} {} </table>'.format(table_header, table_contents)
    return table_ep



class Episode:
    def __init__(self, info, max_len=30):
        self.info = info
        self.maxlen = max_len
        self.video_name = info['video_name']

    def render(self, exp_name=None):
        goal_names = [obj['class_name'] for obj in self.info['target'][1][0]]
        # pdb.set_trace()
        episode_info = 'Episode {}.'.format(self.info['episode'])
        episode_info2 = 'Reward {}. Success {}, Target {}'.format(self.info['reward'], self.info['success'], '_'.join(goal_names))

        result_str = '<h3> {} </h3><br>'.format(episode_info)
        result_str += '<h7> {} </h7><br>'.format(episode_info2)

        titles = ['script_tried', 'script', 'close', 'grabbed']
        obj_close = []
        obj_grabbed = []

        episode = self.info['episode']
        n_steps = len(self.info['script_tried'])
        folder_n = 'tmp_images/' + self.video_name.split('plot')[1]
        folder_imgs = '{}/plots/{}'.format(exp_name, folder_n)
        if not os.path.isdir(folder_imgs):
            os.makedirs(folder_imgs)

        for step in tqdm(range(n_steps)):
            # pdb.set_trace()
            curr_graph = self.info['graph'][step]['graph']
            id2node = {node['id']: node for node in curr_graph['nodes']}
            visible_ids = self.info['visible_ids'][step]
            action_ids = [t for t in self.info['action_ids'][step] if t in visible_ids]
            goal_ids = [node['id'] for node in curr_graph['nodes'] if node['class_name'] in goal_names]

            nodes_close = [id2node[edge['to_id']] for edge in curr_graph['edges'] if edge['from_id'] == 1 and edge['relation_type'] == 'CLOSE' and edge['to_id'] in visible_ids]
            nodes_grabbed = [id2node[edge['to_id']] for edge in curr_graph['edges'] if edge['from_id'] == 1 and 'HOLDS' in edge['relation_type'] and edge['to_id'] in visible_ids]
            close_str = ['{}.{}'.format(node['class_name'], node['id']) for node in nodes_close]
            grabbed_str = ['{}.{}'.format(node['class_name'], node['id']) for node in nodes_grabbed]
            obj_close.append(close_str)
            obj_grabbed.append(grabbed_str)

            # TODO: uncomment this for potting better
            save_graph_2d('{}/plot_{}_{:02d}.png'.format(folder_imgs, episode, step), self.info['graph'][step], visible_ids, action_ids, goal_ids)
        
        
        folder_name = os.path.dirname(self.video_name)
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)
        subprocess.call([
            'ffmpeg', '-framerate', '2', '-i', '{}/plot_{}_%02d.png'.format(folder_imgs, episode), '-r', '30', '-pix_fmt', 'yuv420p',
            '{}.mp4'.format(self.video_name)
        ])

        col_info = [self.info['script_tried'], self.info['script_done'], obj_close, obj_grabbed]

        result_str += html_table(titles, self.maxlen, col_info, [ "width: 15%", "width: 15%", "", ""])
        steps_title = ['step {}'.format(it) for it in range(n_steps)]
        link_table = """<a href=# onclick="toggle_visibility('plot_ep_""" + str(
            self.info['episode']) + """');"> show results </a>"""

        ep_info = [[self.info['script_done'][it], html_img('./plots/plot_{}_{}.png'.format(episode, it))] for it in range(n_steps)]
        result_str += '<br>' + link_table + '<div id="plot_ep_{}" style="display: None">'.format(str(self.info['episode'])) + html_table(steps_title, 2, ep_info) + '</div>'
        # Render the table


        # result_str += """<a href="./plots/ep_"""+str(self.info['episode'])+""".html">First episode </a>"""

        #result_str += render(self.info)
        result_str += '</div><br>'



        #html_3d_plot = render(self.info)
        html_3d_plot = ''

        return result_str, html_3d_plot

class Plotter:
    def __init__(self, experiment_name='test', root_dir=None):
        self.experiment_name = experiment_name
        if root_dir is None:
            print("Root path is missing. Please set root_dir")
            raise 
        else:
            self.root_path = root_dir
        self.dir_name = '{}/plots/{}'.format(self.root_path, self.experiment_name)
        self.episodes = []
        self.index_annot = 0

    def add_episode(self, info):
        self.episodes.append(Episode(info))

    def render(self):
        print("Rendering...")
        if not os.path.isdir(self.dir_name):
            os.makedirs(self.dir_name)

        if not os.path.isdir(self.dir_name+'/plots'):

            os.makedirs(self.dir_name + '/plots')

        file_name = '{}/result.html'.format(self.dir_name)
        file_name2 = '{}/content.html'.format(self.dir_name)

        content_str = """
        <html> 
          <head> 
            <script src="https://cdn.plot.ly/plotly-latest.min.js" charset="utf-8"></script>
            <script src="https://code.jquery.com/jquery-3.5.0.min.js"></script> 
            <script> 
            $(function(){
              $("#includedContent").load("content.html"); 
            });
            </script> 
                        <script type="text/javascript">
            <!--
                function toggle_visibility(id) {
                   var e = document.getElementById(id);
                   if(e.style.display == 'block')
                      e.style.display = 'none';
                   else
                      e.style.display = 'block';
                }
            //-->
            </script>
          </head> 
        
          <body> 
             <h3> """+ self.experiment_name + """ </h3>
             <div id="includedContent"></div>
          </body> 
        </html>    
        """


        plot_html_header = """
          <head> 
            <script src="https://cdn.plot.ly/plotly-latest.min.js" charset="utf-8"></script>
          </head>
        """
        html_str = ''
        while self.index_annot < len(self.episodes):
            episode = self.episodes[self.index_annot]
            content_table, plot_html = episode.render(self.dir_name)
            plot_html_full = '<html>' + plot_html_header + '<body>' + plot_html + '</body></html>'
            html_str += content_table
            with open('{}/plots/ep_{}.html'.format(self.dir_name, self.index_annot), 'w+') as fo:
                fo.writelines(plot_html_full)
            self.index_annot += 1

        with open(file_name, 'w+') as f:
            f.writelines(content_str)

        with open(file_name2, 'a+') as f:
            f.writelines(html_str)

        print("Done")

def plot_single(input_path, out_file, object_classes_remark=[]):
    path = input_path
    # path = '../record_graph/testmeet/task_0/time.09.21.2020-11.21.38/'
    init_graph_file = '{}/init_graph.json'.format(path)
    json_files = glob.glob('{}/file*.json'.format(path))
    
    with open(init_graph_file, 'r') as f:
        init_graph = json.load(f)

    id2nodeinit_graph = {node['id']: node for node in init_graph['graph']['nodes']}

    plot = Plotter(root_dir='./html/')
    graph_list = []
    json_files = sorted(json_files, key=lambda name: int(name.split('file_')[-1].replace('.json', '')))
    with open(json_files[-1], 'r') as f:
        print(input_path)
        ctt = json.load(f)
        print(ctt['time'])
    for json_file in json_files:
        with open(json_file, 'r') as f:
            graph = json.load(f)

            # Add other node info
            for node in graph['graph']['nodes']:
                init_graph_node = id2nodeinit_graph[node['id']]
                node['class_name'] = init_graph_node['class_name']
                node['category'] = init_graph_node['category']


        graph_list.append(graph)
    content = {'graph': graph_list}
    content['target'] = [[[]],[[]]]
    # episode = os.path.splitext(json_file)[0].split('/')[-1].split('_')[1]
    # print(episode, os.path.splitext(json_file)[0], os.path.splitext(json_file)[0].split('/')[-1])
    content['episode'] = 0
    content['reward'] = None
    content['success'] = None
    content['script_tried'] = [None] * len(graph_list)
    content['script_done'] = [None] * len(graph_list)
    ids_curr = [node['id'] for node in init_graph['graph']['nodes'] if node['class_name'] in object_classes_remark]
    
    content['visible_ids'] = [ids_curr,] * len(graph_list)
    content['action_ids'] = [ids_curr, ] * len(graph_list)
    content['goal_ids'] = [[], ] * len(graph_list)
    content['video_name'] = out_file
    plot.add_episode(content)
    plot.render()
    num_steps = len(graph_list)
    return out_file, num_steps

if __name__ == '__main__':
    data_input_file = '../data_input/test_env_set_help_20_neurips.pik'
    with open(data_input_file, 'rb') as f:
        file_inp = pkl.load(f)

    task_id2class_names = {}
    for i in range(100):
        task_id2class_names[i] = []
        for pred, cont in file_inp[i]['goal_class'].items():
            if cont > 0 and '1' not in pred:
                task_id2class_names[i].append(pred.split('_')[1])

    files_inp = glob.glob('../record_graph/test/*/*')
#    print(len(files_inp))
    for file_name in files_inp:
        task_id = re.findall(r"task_+[0-9][0-9]?", file_name)
        task_id = (int)(task_id[0].split('_')[-1])
        input_path = file_name
        vid_out = input_path.replace('../record_graph', '../plots')
#        vid_out = input_path
        print(vid_out)
        name, steps = plot_single(input_path, vid_out, object_classes_remark=task_id2class_names[task_id])
        print(name, steps)
    
