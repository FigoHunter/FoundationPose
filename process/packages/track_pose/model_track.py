from foundation_pose.estimator import *
from foundation_pose.datareader import *
import argparse
from datetime import datetime

class TrackHandle:
  def __init__(self):
    self.scorer = None
    self.refiner = None
    self.glctx = None
    self.est = None

def init_module():
  set_logging_format()
  set_seed(0)
 
  handle = TrackHandle

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)).to_mesh()
  est = FoundationPose(model_pts=mesh_tmp.vertices.copy(), model_normals=mesh_tmp.vertex_normals.copy(), mesh=mesh_tmp, scorer=None, refiner=None, glctx=glctx, debug_dir='./debug', debug=0)

  handle.scorer = scorer
  handle.refiner = refiner
  handle.glctx = glctx
  handle.est = est
  logging.info("estimator initialization done")
  
  return handle


def process(handle, mesh_file, scene_dir, tracked_obj, est_refine_iter=5, track_refine_iter=2, debug=0, debug_dir=None):
  mesh = trimesh.load(mesh_file)
  if debug_dir is None:
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    debug_dir = f'{scene_dir}/debug'
  debug_dir = f'{debug_dir}/{timestamp}'
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  est = handle.est
  est.reset_object(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh)
  logging.info("estimator mesh set")

  reader = YcbineoatReader(video_dir=scene_dir, shorter_side=None, zfar=np.inf)

  for i in range(len(reader.color_files)):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    if i==0:
      mask = reader.get_mask(0,tracked_obj).astype(bool)
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)

      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.1
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=track_refine_iter)
  
    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      cv2.imshow('1', vis[...,::-1])
      cv2.waitKey(1)


    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)