from __future__ import division, print_function

import argparse

import cv2
import numpy as np
import torch

from model import ExpandNet
from smooth import smoothen_luminance
from util import (
    process_path,
    map_range,
    str2bool,
    cv2torch,
    torch2cv,
    tone_map,
    create_tmo_param_from_args, preprocess, create_name,
)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--video', default='input_video.mp4', help='Video to convert')
    arg(
        '--out',
        type=lambda x: process_path(x, True),
        default='results/',
        help='Output location',
    )
    arg(
        '--smooth',
        type=str2bool,
        default=False,
        help='Whether use smooth luminance method',
    )
    arg(
        '--save_hdr_frames',
        type=str2bool,
        default=False,
        help='Whether save all frames in exr format',
    )
    arg(
        '--save_ldr_frames',
        type=str2bool,
        default=False,
        help='Whether save all frames in png format',
    )
    arg(
        '--patch_size',
        type=int,
        default=256,
        help='Patch size (to limit memory use).',
    )
    arg('--resize', type=str2bool, default=False, help='Use resized input.')
    arg(
        '--use_exr',
        type=str2bool,
        default=True,
        help='Produce .EXR instead of .HDR files.',
    )
    arg('--width', type=int, default=1024, help='Image width resizing.')
    arg('--height', type=int, default=512, help='Image height resizing.')
    arg('--tag', default=None, help='Tag for outputs.')
    arg(
        '--use_gpu',
        type=str2bool,
        default=torch.cuda.is_available(),
        help='Use GPU for prediction.',
    )
    arg(
        '--tone_map',
        choices=['exposure', 'reinhard', 'mantiuk', 'drago'],
        default='reinhard',
        help='Tone Map resulting HDR image.',
    )
    arg(
        '--stops',
        type=float,
        default=0.0,
        help='Stops (loosely defined here) for exposure tone mapping.',
    )
    arg(
        '--gamma',
        type=float,
        default=1.0,
        help='Gamma curve value (if tone mapping).',
    )
    arg(
        '--use_weights',
        type=process_path,
        default='net_params/baseRec_tuningRec.pth',
        help='Weights to use for prediction',
    )
    opt = parser.parse_args()
    return opt


def load_pretrained(weights):
    net = ExpandNet()
    net.load_state_dict(
        torch.load(weights, map_location=lambda s, l: s)
    )
    net.eval()
    return net


def create_video(opt):
    if opt.tone_map is None:
        opt.tone_map = 'reinhard'
    net = load_pretrained(opt.use_weights)
    video_file = opt.video
    cap_in = cv2.VideoCapture(video_file)
    fps = cap_in.get(cv2.CAP_PROP_FPS)
    print(f"fps: {fps}")
    n_frames = cap_in.get(cv2.CAP_PROP_FRAME_COUNT)
    predictions = []
    lum_percs = []
    i = 0
    while cap_in.isOpened():
        perc = cap_in.get(cv2.CAP_PROP_POS_FRAMES) * 100 / n_frames
        print('\rConverting video: {0:.2f}%'.format(perc), end='')
        ret, loaded = cap_in.read()
        if loaded is None:
            break
        ldr_input = preprocess(loaded, opt)
        t_input = cv2torch(ldr_input)
        if opt.use_gpu:
            net.cuda()
            t_input = t_input.cuda()
        prediction = torch2cv(net.predict(t_input, opt.patch_size).cpu())
        predictions.append(prediction)
        percs = np.percentile(predictions[-1], (1, 25, 50, 75, 99))
        lum_percs.append(percs)
    print()
    cap_in.release()

    if opt.smooth:
        predictions = smoothen_luminance(predictions, lum_percs)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out_vid_name = create_name(
        video_file, 'prediction', 'avi', opt.out, opt.tag
    )
    out_vid = cv2.VideoWriter(out_vid_name, fourcc, fps, (predictions[0].shape[1], predictions[0].shape[0]))
    for i, pred in enumerate(predictions):
        perc = (i + 1) * 100 / n_frames
        print('\rWriting video: {0:.2f}%'.format(perc), end='')
        if opt.save_hdr_frames:
            hdr_format = 'exr' if opt.use_exr else 'hdr'
            cv2.imwrite(f'{opt.out}/frame_%04d.{hdr_format}' % i, map_range(pred, 0, 1))
        tmo_img = tone_map(
            pred, opt.tone_map, **create_tmo_param_from_args(opt)
        )
        tmo_img = (tmo_img * 255).astype(np.uint8)
        if opt.save_ldr_frames:
            cv2.imwrite(f'{opt.out}/frame_{opt.tone_map}_%04d.png' % i, tmo_img)
        out_vid.write(tmo_img)
    print()
    out_vid.release()


def main():
    opt = get_args()
    create_video(opt)


if __name__ == '__main__':
    main()
