"""

    Created on 15/06/21 10:22 PM 
    @author: Kartik Prabhu

"""
import os

import torch

from datetime import datetime as dt

def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def save_checkpoints(cfg, file_path, epoch_idx, encoder, encoder_solver, decoder, decoder_solver, refiner,
                     refiner_solver, merger, merger_solver, best_iou, best_epoch):
    print('[INFO] %s Saving checkpoint to %s ...' % (dt.now(), file_path))

    if not os.path.exists(file_path):
        os.mkdir(file_path)
        file_path = file_path + epoch_idx+'/'
        if not os.path.exists(file_path):
            os.mkdir(file_path)

    checkpoint = {
        'epoch_idx': epoch_idx,
        'best_iou': best_iou,
        'best_epoch': best_epoch,
        'encoder_state_dict': encoder.state_dict(),
        'encoder_solver_state_dict': encoder_solver.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'decoder_solver_state_dict': decoder_solver.state_dict()
    }

    if cfg.pix2vox_refiner:
        checkpoint['refiner_state_dict'] = refiner.state_dict()
        checkpoint['refiner_solver_state_dict'] = refiner_solver.state_dict()
    if cfg.pix2vox_merger:
        checkpoint['merger_state_dict'] = merger.state_dict()
        checkpoint['merger_solver_state_dict'] = merger_solver.state_dict()

    torch.save(checkpoint,  file_path + 'checkpoint_' + epoch_idx + '.pth')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count