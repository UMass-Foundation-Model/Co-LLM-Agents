import glob
import dominate
from dominate.tags import meta, h3, table, tr, td, th, p, a, img, br, video, source

def html_table(content, header=None):
    curr_table = table()
    if header is not None:
        header_html = tr([th(head) for head in header]) 
        curr_table.add(header_html)
    for ct in content:
        curr_table.add(tr([td(c) for c in ct]))
    return curr_table

tasks = [
0,
1,
20,
21,
40,
41,
80,
81,
]

steps2 = [111, 134 ,119 ,101 ,82 ,86 ,95 ,50]
steps1 = [98, 165, 154, 100, 149, 121, 128, 60]

header_data = ['Task name', 'Steps Planner', 'Video Planner', 'Steps Human', 'Video Human']
table_data = []
for it, task_id in enumerate(tasks):
    file1 = glob.glob('../plots/planner/logs_agent_{}_*'.format(task_id))[0]
    file2 = glob.glob('../plots/collectionv0/task_{}/*.mp4'.format(task_id))[0]
    t1 = steps1[it]
    t2 = steps2[it]

    taskname = file1.split('/')[-1].replace('logs_agent_{}'.format(task_id), '')
    vid1 = video(source(src=file1))
    vid2 = video(source(src=file2))
    vid1.attributes['controls'] = True
    vid2.attributes['controls'] = True
    vid1.attributes['width'] = '500px';
    vid2.attributes['width'] = '500px';
    #vid1 = '<video controls><source src="{}" type="video/mp4"></video>'.format(file1)
    #vid2 = '<video controls><source src="{}" type="video/mp4"></video>'.format(file2)
    table_data.append([taskname, t1, vid1, t2, vid2])


content = html_table(table_data, header=header_data)
with open('result.html', 'w+') as f:
    f.write(str(content))
