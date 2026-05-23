import gc
import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.motion_net import MotionNetwork
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, 'point_cloud'))
            else:
                self.loaded_iter = load_iteration
            print('Loading trained model at iteration {}'.format(self.loaded_iter))
        self.train_cameras = {}
        self.test_cameras = {}
        if os.path.exists(os.path.join(args.source_path, 'transforms.json')):
            print('Found transforms.json file!')
            scene_info = sceneLoadTypeCallbacks['Blender'](args.source_path, args.white_background, args.eval, args=args, preload=args.preload)
        else:
            assert False, 'Could not recognize scene type!'
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, 'input.ply'), 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, 'cameras.json'), 'w') as file:
                json.dump(json_cams, file)
        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)
        original_extent = scene_info.nerf_normalization['radius']
        if original_extent < 0.01:
            self.cameras_extent = 1.0
        else:
            self.cameras_extent = original_extent
        for resolution_scale in resolution_scales:
            print('Loading Training Cameras')
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print('Loading Test Cameras')
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, 'point_cloud', 'iteration_' + str(self.loaded_iter), 'point_cloud.ply'))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, scene_info.binding_data)
        gc.collect()

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, 'point_cloud/iteration_{}'.format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, 'point_cloud.ply'))

    def save_deformed(self, iteration, deformed_xyz, deformed_rotations):
        point_cloud_path = os.path.join(self.model_path, 'point_cloud/iteration_{}'.format(iteration))
        self.gaussians.save_deformed_ply(deformed_xyz, deformed_rotations, os.path.join(point_cloud_path, 'point_cloud_deformed.ply'))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
