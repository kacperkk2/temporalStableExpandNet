import os
import argparse

import PIL
from tensorboardX import SummaryWriter
from PIL import Image
import numpy as np
import torchvision
from tqdm import tqdm
import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader

from ssim import SSIM
from util import (
    slice_gauss,
    map_range,
    cv2torch,
    random_tone_map,
    DirectoryDataset,
    str2bool, execute_tone_map, torch2cv, Reinhard, EvalDataset,
)
from model import ExpandNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size', type=int, default=1, help='Batch size.'
    )
    parser.add_argument(
        '--checkpoint_freq',
        type=int,
        default=1000,
        help='Checkpoint model every x epochs.',
    )
    parser.add_argument(
        '-d', '--data_root_path', default='dataset/', help='Path to hdr data.'
    )
    parser.add_argument(
        '--train_phase',
        choices=['first', 'second'],
        default='second',
        help='First phase trains net reconstruction, second phase trains reconstruction plus stability.',
    )
    parser.add_argument(
        '--stability_loss',
        choices=['l1', 'l2', 'rec', 'ssim'],
        default='l2',
        help='Loss used in stability term (only second phase).',
    )
    parser.add_argument(
        '-s',
        '--save_path',
        default='checkpoints',
        help='Path for checkpointing.',
    )
    parser.add_argument(
        '--use_weights',
        default='net_params/baseRec_tuningRec.pth',
        help='Weights to use',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers.',
    )
    parser.add_argument(
        '--loss_freq',
        type=int,
        default=20,
        help='Report (average) loss every x iterations.',
    )
    parser.add_argument(
        '--use_gpu', type=str2bool, default=True, help='Use GPU for training.'
    )

    return parser.parse_args()


class ExpandNetLoss(nn.Module):
    def __init__(self, loss_lambda=5):
        super(ExpandNetLoss, self).__init__()
        self.similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-20)
        self.l1_loss = nn.L1Loss()
        self.loss_lambda = loss_lambda

    def forward(self, x, y):
        cosine_term = (1 - self.similarity(x, y)).mean()
        return self.l1_loss(x, y) + self.loss_lambda * cosine_term


class StabilityL2Loss(nn.Module):
    def __init__(self):
        super(StabilityL2Loss, self).__init__()
        self.l2_loss = nn.MSELoss()

    def forward(self, x, y):
        return self.l2_loss(x, y)


class StabilityL1Loss(nn.Module):
    def __init__(self):
        super(StabilityL1Loss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, x, y):
        return self.l1_loss(x, y)


def random_transformation(img):
    rotation = np.random.uniform(-1.0, 1.0)
    translation_x = np.random.uniform(-2.0, 2.0)
    translation_y = np.random.uniform(-2.0, 2.0)
    zoom = np.random.uniform(0.97, 1.03)
    shearing = np.random.uniform(-1.0, 1.0)

    output_image = torchvision.transforms.functional.affine(
        cv2torch(img),
        angle=rotation,
        translate=(translation_x, translation_y),
        scale=zoom,
        shear=shearing,
        resample=PIL.Image.BILINEAR
    )
    return torch2cv(output_image)


def transform_rec(original_hdr):
    original_hdr = slice_gauss(original_hdr, crop_size=(384, 384), precision=(0.1, 1))
    original_hdr = cv2.resize(original_hdr, (256, 256))
    original_hdr = map_range(original_hdr)
    original_ldr = execute_tone_map(random_tone_map(), original_hdr)
    return cv2torch(original_ldr), cv2torch(original_hdr)


def transform_stab(original_hdr):
    original_hdr = slice_gauss(original_hdr, crop_size=(384, 384), precision=(0.1, 1))
    original_hdr = cv2.resize(original_hdr, (256, 256))
    original_hdr = map_range(original_hdr)
    original_ldr = execute_tone_map(random_tone_map(), original_hdr)
    perturbed_ldr = random_transformation(original_ldr)
    return cv2torch(original_ldr), cv2torch(original_hdr), cv2torch(perturbed_ldr)


def load_pretrained(weights):
    net = ExpandNet()
    net.load_state_dict(
        torch.load(weights, map_location=lambda s, l: s)
    )
    return net


def train_first_phase(opt):
    model = ExpandNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=7e-5)
    reconstruction_loss = ExpandNetLoss()
    dataset = DirectoryDataset(
        data_root_path=opt.data_root_path,
        preprocess=transform_rec
    )
    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=True,
        drop_last=True,
    )
    if opt.use_gpu:
        model.cuda()
        torch.backends.cudnn.benchmark = True

    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    else:
        print(
            f'WARNING: save_path: {opt.save_path}, already exists. '
            'Checkpoints may be overwritten'
        )
    writer = SummaryWriter(opt.save_path + "/stats")

    for epoch in tqdm(range(1, 10_001), desc='Training (first phase)'):
        sum_loss = 0.0
        count = 0
        for i, (original_ldr, original_hdr) in enumerate(tqdm(loader, desc=f'Epoch {epoch}', position=0, leave=True)):
            if opt.use_gpu:
                original_ldr = original_ldr.cuda()
                original_hdr = original_hdr.cuda()

            original_hdr_prediction = model(original_ldr)
            total_loss = reconstruction_loss(original_hdr_prediction, original_hdr)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            sum_loss += total_loss.item()
            count += 1
                
        writer.add_scalar('total_loss', sum_loss / count, epoch)
        writer.file_writer.flush()

        if (epoch % opt.checkpoint_freq) == 0:
            torch.save(
                model.state_dict(),
                os.path.join(opt.save_path, f'epoch_{epoch}.pth'),
            )
    writer.close()


def prepare_stability_loss(opt):
    if opt.stability_loss == 'l1':
        return StabilityL1Loss(), 0.4
    elif opt.stability_loss == 'rec':
        return ExpandNetLoss(), 0.38
    elif opt.stability_loss == 'ssim':
        return SSIM(data_range=1.0), 0.1
    else:
        return StabilityL2Loss(), 0.8


def train_second_phase(opt):
    model = load_pretrained(opt.use_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=7e-5)
    reconstruction_loss = ExpandNetLoss()
    stability_loss, alpha = prepare_stability_loss(opt)
    dataset = DirectoryDataset(
        data_root_path=opt.data_root_path,
        preprocess=transform_stab
    )
    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=True,
        drop_last=True,
    )
    if opt.use_gpu:
        model.cuda()
        torch.backends.cudnn.benchmark = True

    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    else:
        print(
            f'WARNING: save_path: {opt.save_path}, already exists. '
            'Checkpoints may be overwritten'
        )
    writer = SummaryWriter(opt.save_path + "/stats")

    for epoch in tqdm(range(1, 1_601), desc='Training (second phase)'):
        sum_loss = {'total': 0.0, 'rec': 0.0, 'stab': 0.0}
        count = 0
        for i, (original_ldr, original_hdr, perturbed_ldr) in enumerate(tqdm(loader, desc=f'Epoch {epoch}', position=0, leave=True)):
            if opt.use_gpu:
                original_ldr = original_ldr.cuda()
                original_hdr = original_hdr.cuda()
                perturbed_ldr = perturbed_ldr.cuda()

            original_hdr_prediction = model(original_ldr)
            perturbed_hdr_prediction = model(perturbed_ldr)
            rec_loss = (1.0 - alpha) * reconstruction_loss(original_hdr_prediction, original_hdr)
            stab_loss = alpha * stability_loss(perturbed_hdr_prediction, original_hdr_prediction)
            if opt.stability_loss == 'ssim':
                stab_loss *= -1
            total_loss = rec_loss + stab_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            sum_loss['total'] += total_loss.item()
            sum_loss['rec'] += rec_loss.item()
            sum_loss['stab'] += stab_loss.item()
            count += 1

        writer.add_scalar('total_loss', sum_loss['total'] / count, epoch)
        writer.add_scalar('reconstruction_loss', sum_loss['rec'] / count, epoch)
        writer.add_scalar('stability_loss', sum_loss['stab'] / count, epoch)
        writer.file_writer.flush()

        if (epoch % opt.checkpoint_freq) == 0:
            torch.save(
                model.state_dict(),
                os.path.join(opt.save_path, f'epoch_{epoch}.pth'),
            )
    writer.close()


if __name__ == '__main__':
    opt = parse_args()
    if opt.train_phase == 'first':
        print(
            'INFO: starting first phase of training. \n'
            'Omitting stability loss function'
        )
        train_first_phase(opt)
    elif opt.train_phase == 'second':
        print(
            'INFO: starting second phase of training. \n'
            f'Stability loss function: {opt.stability_loss}.\n'
            f'Continuing model training: {opt.use_weights}.'
        )
        train_second_phase(opt)
