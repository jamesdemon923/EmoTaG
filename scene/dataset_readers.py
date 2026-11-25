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
import glob
from utils.sh_utils import SH2RGB
from utils.audio_utils import get_audio_features
from utils.au_utils import load_openface_au_csv
from utils.emotion_utils import load_emotion_features, emotion_score_from_distribution
from utils.mask_utils import pack_bool_mask
from utils.drive_sanitize import load_drive_params, playback_order, raw_drive
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
    """Load binding information from .npy files."""
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

def readCamerasFromTransforms(path, transformsfile, white_background, extension='.jpg', audio_file='', audio_extractor='wav2vec2', N_views=-1, preload=True, model_center=None, audio_lag=0):
    cam_infos = []
    from utils.audio_features import feature_file
    feature_file(audio_extractor)
    N_views = N_views if audio_file == '' else -1
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fl_x = contents.get('fl_x')
        fl_y = contents.get('fl_y')
        frames = contents['frames'] if N_views is None or N_views <= 0 else contents['frames'][:N_views]
        if audio_file == '':
            aud_features = np.load(os.path.join(path, feature_file(audio_extractor)))
        else:
            aud_features = np.load(audio_file)
        aud_features = torch.from_numpy(aud_features)
        aud_features = aud_features.float().permute(0, 2, 1)
        auds = aud_features
        au_features = loadAUFeatures(path, frames)
        emotion_features = None
        emotion_path = os.path.join(path, 'emotion_features.npy')
        if os.path.exists(emotion_path):
            max_frame_id = max((int(frame.get('timestep_index', idx)) for idx, frame in enumerate(frames)))
            emotion_features = load_emotion_features(emotion_path, required_len=max_frame_id + 1)
        looped = audio_file != '' and auds.shape[0] > len(frames)
        if looped:
            flame_file = os.path.join(path, 'flame_params.npz')
            glitched = set(load_drive_params(flame_file)[1]) if os.path.exists(flame_file) else set()
            loop = [f for f in frames if int(f['timestep_index']) not in glitched]
            frames = [loop[i] for i in playback_order(len(loop), auds.shape[0])]
        ldmks_lips = []
        ldmks_mouth = []
        ldmks_lhalf = []
        for idx, frame in tqdm(enumerate(frames)):
            img_id = frame['timestep_index']
            lms = np.loadtxt(os.path.join(path, 'ori_imgs', str(img_id) + '.lms'))
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
                    talking_dict['face_mask'] = pack_bool_mask(complete_head_mask)
                    talking_dict['hair_mask'] = pack_bool_mask(hair_mask)
                else:
                    raise FileNotFoundError(f'Parsing mask not found: {mask_path}')
            if audio_file == '':
                talking_dict['auds'] = get_audio_features(auds, 2, int(img_id) - int(audio_lag))
                if img_id > auds.shape[0]:
                    print('[warnining] audio feature is too short')
                    break
            else:
                talking_dict['auds'] = get_audio_features(auds, 2, idx - int(audio_lag))
                if idx >= auds.shape[0]:
                    break
            talking_dict['frame_id'] = img_id
            au_idx = min(int(img_id), au_features.shape[0] - 1)
            talking_dict['au_features'] = torch.from_numpy(au_features[au_idx]).float()
            src_idx = int(img_id) if audio_file == '' else idx
            win = np.clip(np.arange(src_idx - 4, src_idx + 4), 0, au_features.shape[0] - 1)
            if audio_file != '':
                win = np.clip(np.arange(au_idx - 4, au_idx + 4), 0, au_features.shape[0] - 1)
                if looped and not raw_drive():
                    play = np.clip(np.arange(idx - 4, idx + 4), 0, len(frames) - 1)
                    win = np.clip([int(frames[j]['timestep_index']) for j in play], 0, au_features.shape[0] - 1)
            talking_dict['au_window'] = torch.from_numpy(au_features[win]).float()
            talking_dict['window_frame_ids'] = [int(x) for x in np.arange(src_idx - 4, src_idx + 4)]
            if emotion_features is not None:
                emo_idx = min(int(img_id), emotion_features.shape[0] - 1)
                p_emo = emotion_features[emo_idx]
                talking_dict['emotion'] = torch.from_numpy(p_emo).float()
                talking_dict['emotion_score'] = float(emotion_score_from_distribution(p_emo))
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
            normal_path_candidates = glob.glob(os.path.join(path, 'sapiens/normal/sapiens_*')) if N_views > 0 else []
            depth_path_candidates_ = glob.glob(os.path.join(path, 'sapiens/depth/sapiens_*')) if N_views > 0 else []
            if N_views > 0 and not (normal_path_candidates and depth_path_candidates_) and idx == 0:
                print(f'[data] no Sapiens priors under {path}/sapiens: geometric loss disabled for this scene '
                      f'(run tools/run_sapiens_priors.sh for adaptation scenes).')
            if N_views > 0 and normal_path_candidates and depth_path_candidates_:
                normal_path_candidates.sort(reverse=True)
                normal_path = os.path.join(normal_path_candidates[0], str(img_id) + '.npy')
                talking_dict['normal_path'] = normal_path
                depth_path_candidates = glob.glob(os.path.join(path, 'sapiens/depth/sapiens_*'))
                depth_path_candidates.sort(reverse=True)
                depth_path = os.path.join(depth_path_candidates[0], str(img_id) + '.npy')
                talking_dict['depth_path'] = depth_path
                if preload:
                    normal = np.load(normal_path)
                    talking_dict['normal'] = torch.as_tensor(normal).permute(2, 0, 1)
                    depth = np.load(depth_path)
                    talking_dict['depth'] = torch.as_tensor(depth)
            FovX = focal2fov(fl_x, w)
            FovY = focal2fov(fl_y, h)
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=w, height=h, background=bg, talking_dict=talking_dict))
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension='.jpg', args=None, preload=True, all_for_train=False):
    model_center = getattr(args, 'model_center', None)
    if model_center is None:
        sys.exit(1)
    if not eval:
        print('Reading Training Transforms')
        train_cam_infos = readCamerasFromTransforms(path, 'transforms.json', white_background, extension, args.audio_file, args.audio_extractor, args.N_views, preload=preload, model_center=model_center, audio_lag=getattr(args, 'audio_lag', 0))
    print('Reading Evaluation Transforms')
    test_cam_infos = readCamerasFromTransforms(path, 'transforms.json', white_background, extension, args.audio_file, args.audio_extractor, -1, preload=preload, model_center=model_center, audio_lag=getattr(args, 'audio_lag', 0))
    if all_for_train:
        train_cam_infos.extend(test_cam_infos)
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
