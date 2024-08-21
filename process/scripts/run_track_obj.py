import argparse
import os
import json
from ycb_objects import load as ycb_load

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
args = parser.parse_args()

from track_pose import model_track


jobpath = args.dir

handle = model_track.init_module()


# for job in os.listdir(dir):
#     jobpath = os.path.join(dir, job)
#     if not os.path.isdir(jobpath):
#         continue
#     print(f"Processing job {job}")
task_info= json.load(open(os.path.join(jobpath, "task.json")))
tracked_objects = task_info['track']
for obj in tracked_objects:
    print(f"----Processing object {obj}")
    mesh_file = ycb_load.get_google16k_mesh(obj,'textured.obj')
    model_track.process(handle, mesh_file, jobpath, obj, debug=2)



  