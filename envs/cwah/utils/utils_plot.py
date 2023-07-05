import json
import os

import pdb
import plotly.graph_objs as go

import plotly.io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import plotly.offline
import random
import graphviz

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


def create_cube(n, color='lightpink', opacity=0.1, cont=False):
    c, b = n['bounding_box']['center'], n['bounding_box']['size']

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


def add_box(nodes, args_rect):
    rectangles = []
    centers = [[], []]
    for node in nodes:
        cx, cy = node['bounding_box']['center'][0], node['bounding_box']['center'][2]
        w, h = node['bounding_box']['size'][0], node['bounding_box']['size'][2]
        minx, miny = cx - w / 2., cy - h / 2.
        centers[0].append(cx)
        centers[1].append(cy)
        if args_rect is not None:
            rectangles.append(
                Rectangle((minx, miny), w, h, **args_rect)
            )
    return rectangles, centers


def add_boxes(nodes, ax, points=None, rect=None):
    rectangles = []
    rectangles_class, center = add_box(nodes, rect)
    rectangles += rectangles_class
    if points is not None:
        ax.scatter(center[0], center[1], **points)
    if rect is not None and len(rectangles) > 0:
        ax.add_patch(rectangles[0])
        collection = PatchCollection(rectangles, match_original=True)
        ax.add_collection(collection)


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
    add_boxes(rooms, ax, points=None, rect={'alpha': 0.1})
    add_boxes(container_and_surface, ax, points=None, rect={'fill': False,
                                                                        'edgecolor': 'blue', 'alpha': 0.3})
    add_boxes([char_node], ax, points=None, rect={'facecolor': 'yellow', 'edgecolor': 'yellow', 'alpha': 0.7})
    add_boxes(visible_nodes, ax, points={'s': 2.0, 'alpha': 1.0}, rect={'fill': False,
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
    fig = plot_graph_2d(graph, char_id, visible_ids, action_ids, goal_ids)
    # plt.axis('scaled')
    # print([node['bounding_box']['size'] for node in graph['nodes'] if node['id'] == 1][0])
    fig.tight_layout()


    fig.savefig(img_name)
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
        for step in range(n_steps):
            curr_graph = self.info['graph'][step]
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
            save_graph_2d('{}/plots/plot_{}_{}.png'.format(exp_name, episode, step), self.info['graph'][step], visible_ids, action_ids, goal_ids)

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
            self.root_path = '.'
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


def delete_redundant_edges_and_ids(graph):
    class_nodes_delete = ['wall', 'floor', 'ceiling', 'door', 'curtain', 'window', 'doorjamb']
    ids_delete = [x['id'] for x in graph['nodes'] if x['class_name'] in class_nodes_delete]

    graph['nodes'] = [x for x in graph['nodes'] if x['id'] not in ids_delete]
    graph['edges'] = [x for x in graph['edges'] if x['from_id'] not in ids_delete and x['to_id'] not in ids_delete]

    parent_node = {}
    children_node = {}
    for it, edge in enumerate(graph['edges']):
        if edge['relation_type'] == 'INSIDE':
            if edge['to_id'] not in parent_node.keys(): parent_node[edge['to_id']] = []
            parent_node[edge['to_id']].append(edge['from_id'])
            if edge['from_id'] not in children_node.keys(): children_node[edge['from_id']] = []
            children_node[edge['from_id']].append(edge['to_id'])

    final_edges = []
    for edge in graph['edges']:
        if edge['relation_type'] == 'INSIDE':
            all_parents = children_node[edge['from_id']]
            all_children = parent_node[edge['to_id']]
            if len(set(all_parents).intersection(all_children)) > 0:
                continue
        # else:
        #     if ((edge['from_id'] in children_node and edge['to_id'] in children_node[edge['from_id']]) or
        #             (edge['to_id'] in children_node and edge['from_id'] in children_node[edge['to_id']])):
        #         continue

        final_edges.append(edge)
    graph['edges'] = final_edges
    return graph

class BeliefPlotter():
    def __init__(self, graph, belief):
        self.graph = graph
        self.belief = belief
        self.id2node = {node['id']: node for node in self.graph['nodes']}
        self.map_position = {}

    def sample_position(self, id_child, id_parent):
        # Check parent bounds
        tuple_position = (id_child, id_parent)
        if tuple_position in self.map_position:
            return self.map_position[tuple_position]
        else:
            parent_bounds = self.id2node[id_parent]['bounding_box']
            x = (random.random() - 0.5) * parent_bounds['size'][0] + parent_bounds['center'][0]
            y = (random.random() - 0.5) * parent_bounds['size'][2] + parent_bounds['center'][2]
            self.map_position[tuple_position] = [x,y]
            # pdb.set_trace()
            return [x,y]

    def is_static(self, node):
        return node['id'] in list(self.belief.keys())

    def getclass(self, node):
        return '{}\n{}'.format(node['class_name'], node['id'])
    
    def graph2im(self, graph, special_nodes={}):
        """
        Outputs an image given a graph
        :param graph:
        :return:
        """

        # graph = delete_redundant_edges_and_ids(graph)
        old_graph = graph
        graph = {}

        class_nodes_delete = ['wall', 'floor', 'ceiling', 'door', 'curtain', 'wallpictureframe', 
                              'clothespile', 'closetdrawer', 'hanger', 'rug', 'curtains', 'lightswitch', 'ceilinglamp', 'towel']
        ids_delete = [x['id'] for x in old_graph['nodes'] if x['class_name'] in class_nodes_delete]

        graph['nodes'] = [x for x in old_graph['nodes'] if x not in ids_delete]
        graph['edges'] = [x for x in old_graph['edges'] if x['from_id'] not in ids_delete and x['to_id'] not in ids_delete]

        id2node = {x['id']: x for x in graph['nodes']}

        g = graphviz.Digraph(engine='fdp')
        g.attr(compound='true')

        # g.attr(rank='min')

        container_nodes = list(set([x['to_id'] for x in graph['edges'] if x['relation_type'] == 'INSIDE']))
        children = {}
        parent = {}
        num_children_subgraph = {}
        for edge in graph['edges']:
            if edge['relation_type'] == 'INSIDE':
                if edge['to_id'] not in children:
                    children[edge['to_id']] = []
                    num_children_subgraph[edge['to_id']] = 0
                parent[edge['from_id']] = edge['to_id']
                children[edge['to_id']].append(edge['from_id'])
                if edge['from_id'] in container_nodes:
                    num_children_subgraph[edge['to_id']] += 1

        subgraphs_added = {}

        curr_subgraphs = [x for (x,y) in num_children_subgraph.items() if y == 0]
        # pdb.set_trace()
        #
        while len(curr_subgraphs) > 0:
            next_subgraphs = []
            for curr_subgraph_id in curr_subgraphs:
                name_graph = self.getclass(id2node[curr_subgraph_id])
                cng = graphviz.Digraph(name='cluster_'+str(curr_subgraph_id))
                cng.attr(label=name_graph)
                cng.node(name=str(curr_subgraph_id), style='invis')
                # pdb.set_trace()
                children_c_graph = children[curr_subgraph_id]
                for child in children_c_graph:
                    if self.is_static(id2node[child]):
                        position = [id2node[child]['bounding_box']['center'][coord_axis] for coord_axis in [0,2]]
                    else:
                        position = self.sample_position(child, curr_subgraph_id)
                    position = ",".join([str(coord/2000.) for coord in position])
                    position += "!"
                    # import pdb
                    # pdb.set_trace()
                    if child in subgraphs_added:
                        cng.subgraph(subgraphs_added[child])

                    else:
                        color_special = {
                            'agent': 'darkseagreen',
                            'goal': 'lightblue'
                        }
                        if child in special_nodes:
                            color_sp = color_special[special_nodes[child]]
                            cng.node(name=str(child), label=self.getclass(id2node[child]), color=color_sp, style='filled', pos=position)

                        else:
                            cng.node(name=str(child), label=self.getclass(id2node[child]), pos=position)

                if id2node[curr_subgraph_id]['category'] == 'Rooms':
                    coords = id2node[curr_subgraph_id]['bounding_box']
                    positionsx = [-1, -1, 1, 1]
                    positionsy = [-1, 1, -1, 1]
                    positionsrx = [coords['center'][0] + px/2. * coords['size'][0] for px in positionsx]
                    positionsry = [coords['center'][2] + py/2. * coords['size'][2] for py in positionsy]
                    for itp, (px,py) in enumerate(zip(positionsrx, positionsry)):
                        cng.node(name='{}.{}'.format(curr_subgraph_id, itp), label='{}.{}'.format(curr_subgraph_id, itp), pos='{},{}!'.format(px/2000., py/2000.))

                subgraphs_added[curr_subgraph_id] = cng
                if curr_subgraph_id == 1:
                    pdb.set_trace()
                if curr_subgraph_id not in parent:
                    g.subgraph(cng)
                    continue
                parent_graph = parent[curr_subgraph_id]
                num_children_subgraph[parent_graph] -= 1
                if num_children_subgraph[parent_graph] == 0:
                    next_subgraphs.append(parent_graph)
            curr_subgraphs = next_subgraphs

        colors = {
            'INSIDE': 'yellow',
            'ON': 'blue',
            'CLOSE': 'purple',
            'CLOSE_CHAR': 'orange',
            'FACING': 'red',
            'BETWEEN': 'green',
            'CLOSEg': 'blue',
            'HOLDS_R': 'cyan',
            'HOLDS_L': 'cyan',

        }
        style = {
            'INSIDE': '',
            'ON': '',
            'HOLDS_R': '',
            'HOLDS_L': '',
            'CLOSE': 'invis',
            'CLOSEg': 'invis',
            'FACING': 'invis',
            'BETWEEN': 'invis',
            'CLOSE_CHAR': ''
        }
        print('Edges...') #
        max_num = 0.2
        id_char = [x for x,y in special_nodes.items() if y == 'agent']
        if len(id_char) > 0:
            id_char = id_char[0]

        # Add close edges
        virtual_edges = []
        rooms = [x['id'] for x in graph['nodes'] if x['category'] == 'Rooms']
        for room_id in rooms:
            if room_id not in children:
                continue
            children_nodes = children[room_id]
            from_node = random.choices(children_nodes, k=6)
            to_node = random.choices(children_nodes, k=6)
            for from_id, to_id in zip(from_node, to_node):
                virtual_edges.append({'from_id': from_id, 'to_id': to_id, 'relation_type': 'CLOSEg'})

        for edge in graph['edges']+virtual_edges:
            rt = edge['relation_type']
            if rt != 'INSIDE' and edge['from_id'] not in ids_delete and edge['to_id'] not in ids_delete:
                if rt == 'CLOSE':
                    # print(edge['from_id'], edge['to_id'], edge['relation_type'])
                    if edge['from_id'] == id_char:
                        rt = 'CLOSE_CHAR'
                        print(rt)
                    else:
                        if parent[edge['from_id']] != parent[edge['to_id']]:
                            continue
                        if random.random() < max_num:
                            continue
                # if ((edge['to_id'] in parent and parent[edge['to_id']] == edge['from_id']) or
                #    (edge['from_id'] in parent and parent[edge['from_id']] == edge['to_id'])):
                #     continue

                if edge['from_id'] not in children and edge['to_id'] not in children:
                    g.edge(str(edge['from_id']), str(edge['to_id']), color=colors[rt], style=style[rt])

                elif edge['from_id'] not in children:
                    g.edge(str(edge['from_id']), str(edge['to_id']), color=colors[rt],
                           lhead='cluster_' + str(edge['to_id']), style=style[rt])
                elif edge['to_id'] not in children:
                    g.edge(str(edge['from_id']), str(edge['to_id']), color=colors[rt],
                           ltail='cluster_' + str(edge['from_id']), style=style[rt])
                else:
                    g.edge(str(edge['from_id']), str(edge['to_id']), color=colors[rt],
                           ltail='cluster_'+str(edge['from_id']), lhead='cluster_'+str(edge['to_id']), style=style[rt])
        return g


if __name__ == '__main__':
    # with open('trained_models//env.virtualhome/task.find-numproc.1-obstype.full-sim.unity/taskset.setup_table/mode.RL-algo.a2c-base.TF-gamma.0.95-cclose.0.0-cgoal.0.0-lr0.0001debug/log.json', 'r') as f:
        # content = json.load(f)
    plot = Plotter()
    # plot.add_episode(content)
    plot.render()
    # pdb.set_trace()