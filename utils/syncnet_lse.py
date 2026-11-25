"""Sync-C / Sync-E (LSE-C / LSE-D) with the public SyncNet model."""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

FPS, SR, VSHIFT = 25, 16000, 15
FACEDET_SCALE, FACEDET_CONF, CROP_SCALE = 0.25, 0.9, 0.40
NUM_FAILED_DET, MIN_FACE_SIZE = 25, 100


def _iou(a, b) -> float:
    xa, ya, xb, yb = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def read_frames(path: str) -> np.ndarray:
    import imageio.v2 as imageio
    return np.stack([f[..., :3] for f in imageio.mimread(path, memtest=False)])


def read_audio_int16(path: str) -> np.ndarray:
    """Mono 16 kHz int16 audio of a wav/video file (via ffmpeg)."""
    cmd = ['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(SR), '-f', 's16le', '-']
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.int16)


class SyncNetScorer:
    def __init__(self, syncnet_root: str | None = None, device: str = 'cuda'):
        import torch
        root = syncnet_root or os.environ.get('SYNCNET_ROOT')
        if not root:
            raise FileNotFoundError('Set SYNCNET_ROOT to a joonson/syncnet_python checkout (with downloaded weights).')
        weights = os.path.join(root, 'data', 'syncnet_v2.model')
        s3fd_w = os.path.join(root, 'detectors', 's3fd', 'weights', 'sfd_face.pth')
        for p in (weights, s3fd_w):
            if not os.path.exists(p):
                raise FileNotFoundError(f'missing {p} (run download_model.sh in {root})')
        if root not in sys.path:
            sys.path.insert(0, root)
        from SyncNetModel import S
        import detectors.s3fd as s3fd_module
        s3fd_module.PATH_WEIGHT = s3fd_w
        self.device = device
        self.net = S(num_layers_in_fc_layers=1024).to(device)
        state = torch.load(weights, map_location='cpu', weights_only=True)
        own = self.net.state_dict()
        for name, param in state.items():
            own[name].copy_(param)
        self.net.eval()
        self.detector = s3fd_module.S3FD(device=device)

    def _track(self, frames, min_track):
        from scipy.interpolate import interp1d
        dets = []
        for fidx, image in enumerate(frames):
            boxes = self.detector.detect_faces(image, conf_th=FACEDET_CONF, scales=[FACEDET_SCALE])
            dets.append([{'frame': fidx, 'bbox': b[:-1].tolist()} for b in boxes])
        tracks = []
        while True:
            track = []
            for framefaces in dets:
                for face in list(framefaces):
                    if not track:
                        track.append(face); framefaces.remove(face)
                    elif face['frame'] - track[-1]['frame'] <= NUM_FAILED_DET:
                        if _iou(face['bbox'], track[-1]['bbox']) > 0.5:
                            track.append(face); framefaces.remove(face); continue
                    else:
                        break
            if not track:
                break
            if len(track) > min_track:
                fn = np.array([f['frame'] for f in track])
                bb = np.array([f['bbox'] for f in track])
                fi = np.arange(fn[0], fn[-1] + 1)
                bbi = np.stack([interp1d(fn, bb[:, j])(fi) for j in range(4)], 1)
                if max(np.mean(bbi[:, 2] - bbi[:, 0]), np.mean(bbi[:, 3] - bbi[:, 1])) > MIN_FACE_SIZE:
                    tracks.append({'frame': fi, 'bbox': bbi})
        return tracks

    @staticmethod
    def _crop(frames, track):
        import cv2
        from scipy import signal
        s = np.array([max(b[3] - b[1], b[2] - b[0]) / 2 for b in track['bbox']])
        y = np.array([(b[1] + b[3]) / 2 for b in track['bbox']])
        x = np.array([(b[0] + b[2]) / 2 for b in track['bbox']])
        k = 13 if len(s) >= 13 else max(1, (len(s) // 2) * 2 - 1)
        if k >= 3:
            s, x, y = signal.medfilt(s, k), signal.medfilt(x, k), signal.medfilt(y, k)
        out, cs = [], CROP_SCALE
        for i, f in enumerate(track['frame']):
            bs = s[i]
            bsi = int(bs * (1 + 2 * cs))
            img = np.pad(frames[f], ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
            my, mx = y[i] + bsi, x[i] + bsi
            face = img[int(my - bs):int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
            out.append(cv2.resize(face, (224, 224)))
        return np.stack(out)

    def score(self, video_path: str, audio_path: str, batch_size: int = 20, audio_start_frame: int = 0) -> dict:
        """audio_start_frame."""
        import torch
        import python_speech_features
        frames = read_frames(video_path)
        audio = read_audio_int16(audio_path)[int(round(audio_start_frame / FPS * SR)):]
        tracks = self._track(frames, min_track=min(100, len(frames) // 2))
        if not tracks:
            return {'status': 'no_face_track'}
        track = max(tracks, key=lambda t: len(t['frame']))
        crops = self._crop(frames, track)[..., ::-1]
        im = np.transpose(np.stack(list(crops), axis=3)[None], (0, 3, 4, 1, 2))
        imtv = torch.from_numpy(np.ascontiguousarray(im).astype(np.float32))
        start = int(round(track['frame'][0] / FPS * SR))
        audio = audio[start:start + int(round(len(crops) / FPS * SR))]
        mfcc = np.stack([np.array(i) for i in zip(*python_speech_features.mfcc(audio, SR))])
        cct = torch.from_numpy(mfcc[None, None].astype(np.float32))
        last = min(len(crops), int(np.floor(len(audio) / 640))) - 5
        imf, ccf = [], []
        with torch.no_grad():
            for i in range(0, last, batch_size):
                rng = range(i, min(last, i + batch_size))
                imf.append(self.net.forward_lip(torch.cat([imtv[:, :, v:v + 5] for v in rng]).to(self.device)).cpu())
                ccf.append(self.net.forward_aud(torch.cat([cct[:, :, :, v * 4:v * 4 + 20] for v in rng]).to(self.device)).cpu())
        imf, ccf = torch.cat(imf), torch.cat(ccf)
        win = VSHIFT * 2 + 1
        ccp = torch.nn.functional.pad(ccf, (0, 0, VSHIFT, VSHIFT))
        d = torch.stack([torch.nn.functional.pairwise_distance(imf[[i]].repeat(win, 1), ccp[i:i + win]) for i in range(len(imf))], 1)
        md = d.mean(1)
        mn, idx = md.min(0)
        return {'status': 'ok', 'Sync-C': float(md.median() - mn), 'Sync-E': float(mn),
                'offset': int(VSHIFT - int(idx)), 'track_frames': int(len(track['frame']))}
