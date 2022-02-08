from __future__ import division, print_function

import argparse
import os

import cv2
import torch

from model import ExpandNet
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
    arg('--ldr', default='input_images/', help='Directory with ldr image(s)')
    arg(
        '--out',
        type=lambda x: process_path(x, True),
        default='results/',
        help='Output location',
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
    arg(
        '--ldr_extensions',
        nargs='+',
        type=str,
        default=['.jpg', '.jpeg', '.tiff', '.bmp', '.png'],
        help='Allowed LDR image extensions',
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


def create_images(opt):
    net = load_pretrained(opt.use_weights)
    # Treat this as a directory of ldr images
    opt.ldr = [
        os.path.join(opt.ldr, f)
        for f in os.listdir(opt.ldr)
        if any(f.lower().endswith(x) for x in opt.ldr_extensions)
    ]
    for ldr_file in opt.ldr:
        loaded = cv2.imread(
            ldr_file, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR
        )
        if loaded is None:
            print('Could not load {0}'.format(ldr_file))
            continue
        ldr_input = preprocess(loaded, opt)
        if opt.resize:
            out_name = create_name(
                ldr_file, 'resized', 'jpg', opt.out, opt.tag
            )
            cv2.imwrite(out_name, (ldr_input * 255).astype(int))

        t_input = cv2torch(ldr_input)
        if opt.use_gpu:
            net.cuda()
            t_input = t_input.cuda()
        prediction = map_range(
            torch2cv(net.predict(t_input, opt.patch_size).cpu()), 0, 1
        )
        extension = 'exr' if opt.use_exr else 'hdr'
        out_name = create_name(
            ldr_file, 'prediction', extension, opt.out, opt.tag
        )
        print(f'Writing {out_name}')
        cv2.imwrite(out_name, prediction)
        if opt.tone_map is not None:
            tmo_img = tone_map(
                prediction, opt.tone_map, **create_tmo_param_from_args(opt)
            )
            out_name = create_name(
                ldr_file,
                'prediction_{0}'.format(opt.tone_map),
                'jpg',
                opt.out,
                opt.tag,
            )
            cv2.imwrite(out_name, (tmo_img * 255).astype(int))


def main():
    opt = get_args()
    create_images(opt)


if __name__ == '__main__':
    main()
