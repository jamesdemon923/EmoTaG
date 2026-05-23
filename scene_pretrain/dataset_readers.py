import os
import sys
import torch
from PIL import Image
from typing import NamedTuple, Dict
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import pandas as pd
import pdb
from utils.sh_utils import SH2RGB
from utils.audio_utils import get_audio_features
from utils.au_utils import load_openface_au_csv
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    background: np.array
    talking_dict: dict

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    binding_data: Dict = None

def getNerfppNorm(cam_info):

    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return (center.flatten(), diagonal)
    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center
    return {'translate': translate, 'radius': radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def fetchBindingData(path):
    """
    Load binding information from .npy files.
    """
    try:
        face_indices = np.load(os.path.join(path, 'face_indices.npy'))
        bary_coords = np.load(os.path.join(path, 'bary_coords.npy'))
        vertices = np.load(os.path.join(path, 'vertices.npy'))
        return {'face_indices': face_indices, 'bary_coords': bary_coords, 'vertices': vertices}
    except FileNotFoundError:
        return None

def storePly(path, xyz, rgb):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def loadAUFeatures(path, frames, au_dim=6):
    max_frame_id = max((int(frame.get('timestep_index', idx)) for idx, frame in enumerate(frames)))
    required_len = max_frame_id + 1
    au_path = os.path.join(path, 'au_features.csv')
    if os.path.exists(au_path):
        au_features = load_openface_au_csv(au_path, required_len=required_len)
        if au_features.ndim != 2 or au_features.shape[1] != au_dim:
            raise ValueError(f'au_features.csv must provide shape [N, {au_dim}], got {au_features.shape}.')
        return au_features
    raise FileNotFoundError(f'Missing AU feature file: {au_path}')

def readCamerasFromTransforms(path, transformsfile, white_background, extension='.jpg', audio_file='', audio_extractor='wav2vec2', preload=True, model_center=None):
    cam_infos = []
    if audio_extractor != 'wav2vec2':
        raise ValueError('EmoTaG expects --audio_extractor wav2vec2.')
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fl_x = contents.get('fl_x')
        fl_y = contents.get('fl_y')
        frames = contents['frames']
        if audio_file == '':
            aud_features = np.load(os.path.join(path, 'aud_w2v.npy'))
        else:
            aud_features = np.load(audio_file)
        aud_features = torch.from_numpy(aud_features)
        aud_features = aud_features.float().permute(0, 2, 1)
        auds = aud_features
        au_features = loadAUFeatures(path, frames)
        ldmks_lips = []
        ldmks_mouth = []
        ldmks_lhalf = []
        for idx, frame in tqdm(enumerate(frames)):
            img_id = frame['timestep_index']
            lms_path = os.path.join(path, 'ori_imgs', str(img_id) + '.lms')
            lms = np.loadtxt(lms_path)
            lips = slice(48, 60)
            mouth = slice(60, 68)
            xmin, xmax = (int(lms[lips, 1].min()), int(lms[lips, 1].max()))
            ymin, ymax = (int(lms[lips, 0].min()), int(lms[lips, 0].max()))
            ldmks_lips.append([int(xmin), int(xmax), int(ymin), int(ymax)])
            ldmks_mouth.append([int(lms[mouth, 1].min()), int(lms[mouth, 1].max())])
            lh_xmin, lh_xmax = (int(lms[31:36, 1].min()), int(lms[:, 1].max()))
            xmin, xmax = (int(lms[:, 1].min()), int(lms[:, 1].max()))
            ymin, ymax = (int(lms[:, 0].min()), int(lms[:, 0].max()))
            ldmks_lhalf.append([lh_xmin, lh_xmax, ymin, ymax])
        ldmks_lips = np.array(ldmks_lips)
        ldmks_mouth = np.array(ldmks_mouth)
        ldmks_lhalf = np.array(ldmks_lhalf)
        mouth_lb = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).min()
        mouth_ub = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).max()
        for idx, frame in tqdm(enumerate(frames)):
            img_id = frame['timestep_index']
            cam_name = os.path.join(path, 'gt_imgs', str(img_id) + extension)
            c2w = np.array(frame['transform_matrix'])
            c2w[:3, 1:3] *= -1
            c2w[:3, 3] *= 1 / 2.4
            if model_center is not None:
                c2w[:3, 3] += model_center
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]
            talking_dict = {}
            talking_dict['img_id'] = img_id
            image_path = cam_name
            image_name = Path(cam_name).stem
            talking_dict['image_path'] = image_path
            if preload or idx == 0:
                image = Image.open(image_path)
                w, h = (image.size[0], image.size[1])
                image = np.array(image.convert('RGB'))
            torso_img_path = os.path.join(path, 'torso_imgs', str(img_id) + '.png')
            bg_img_path = os.path.join(path, 'bc.jpg')
            talking_dict['torso_img_path'] = torso_img_path
            talking_dict['bg_img_path'] = bg_img_path
            if preload:
                torso_img = np.array(Image.open(torso_img_path).convert('RGBA')) * 1.0
                bg_img = np.array(Image.open(bg_img_path).convert('RGB'))
                bg = torso_img[..., :3] * torso_img[..., 3:] / 255.0 + bg_img * (1 - torso_img[..., 3:] / 255.0)
                bg = bg.astype(np.uint8)
            if not preload:
                image = bg = None
            teeth_mask_path = os.path.join(path, 'teeth_mask', str(img_id) + '.npy')
            mask_path = os.path.join(path, 'parsing', str(img_id) + '.png')
            talking_dict['teeth_mask_path'] = teeth_mask_path
            talking_dict['mask_path'] = mask_path
            if preload:
                if os.path.exists(mask_path):
                    mask = np.array(Image.open(mask_path).convert('RGB')) * 1.0
                    hair_mask = (mask[:, :, 0] < 1) & (mask[:, :, 1] < 1) & (mask[:, :, 2] < 1)
                    skin_mask = (mask[:, :, 2] > 254) & (mask[:, :, 0] == 0) & (mask[:, :, 1] == 0)
                    mouth_interior_mask = (mask[:, :, 0] == 100) & (mask[:, :, 1] == 100) & (mask[:, :, 2] == 100)
                    neck_mask = (mask[:, :, 0] == 0) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 0)
                    complete_head_mask = hair_mask | skin_mask | mouth_interior_mask | neck_mask
                    talking_dict['face_mask'] = complete_head_mask
                    talking_dict['hair_mask'] = hair_mask
                    face_mask_no_mouth = hair_mask | skin_mask | neck_mask
                    talking_dict['face_mask_no_mouth'] = face_mask_no_mouth
                    talking_dict['mouth_interior_mask'] = mouth_interior_mask
                else:
                    raise FileNotFoundError(f'Parsing mask not found: {mask_path}')
            if audio_file == '':
                talking_dict['auds'] = get_audio_features(auds, 2, img_id)
                if img_id > auds.shape[0]:
                    print('[warnining] audio feature is too short')
                    break
            else:
                talking_dict['auds'] = get_audio_features(auds, 2, idx)
                if idx >= auds.shape[0]:
                    break
            talking_dict['frame_id'] = img_id
            au_idx = min(int(img_id), au_features.shape[0] - 1)
            talking_dict['au_features'] = torch.from_numpy(au_features[au_idx]).float()
            [xmin, xmax, ymin, ymax] = ldmks_lips[idx].tolist()
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2
            l = max(xmax - xmin, ymax - ymin) // 2
            xmin = cx - l
            xmax = cx + l
            ymin = cy - l
            ymax = cy + l
            talking_dict['lips_rect'] = [xmin, xmax, ymin, ymax]
            talking_dict['lhalf_rect'] = ldmks_lhalf[idx]
            talking_dict['mouth_bound'] = [mouth_lb, mouth_ub, ldmks_mouth[idx, 1] - ldmks_mouth[idx, 0]]
            talking_dict['img_id'] = img_id
            FovX = focal2fov(fl_x, w)
            FovY = focal2fov(fl_y, h)
            if idx > 5000:
                break
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=w, height=h, background=None, talking_dict=talking_dict))
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension='.jpg', args=None, preload=True):
    model_center = getattr(args, 'model_center', None)
    if model_center is None:
        sys.exit(1)
    if not eval:
        print('Reading Training Transforms')
        train_cam_infos = readCamerasFromTransforms(path, 'transforms.json', white_background, extension, args.audio_file, args.audio_extractor, preload=preload, model_center=model_center)
    print('Reading Evaluation Transforms')
    test_cam_infos = readCamerasFromTransforms(path, 'transforms.json', white_background, extension, args.audio_file, args.audio_extractor, preload=preload, model_center=model_center)
    if eval:
        train_cam_infos = test_cam_infos
    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, 'points3D.ply')
    point_cloud = None
    binding_data = None
    if os.path.exists(ply_path):
        point_cloud = fetchPly(ply_path)
        binding_data = fetchBindingData(path)
        if binding_data:
            print('Loaded FLAME binding data.')
        else:
            print('Missing FLAME binding data.')
            sys.exit(1)
    if point_cloud is None:
        sys.exit(1)
    scene_info = SceneInfo(point_cloud=point_cloud, train_cameras=train_cam_infos, test_cameras=test_cam_infos, nerf_normalization=nerf_normalization, ply_path=ply_path, binding_data=binding_data)
    return scene_info
sceneLoadTypeCallbacks = {'Colmap': None, 'Blender': readNerfSyntheticInfo}
