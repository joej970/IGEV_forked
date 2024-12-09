import sys
sys.path.append('core')
import argparse
import os

from pathlib import Path

from igev_stereo import IGEVStereo
from prettytable import PrettyTable
from utils.utils import InputPadder

import numpy as np
from PIL import Image
import torch
DEVICE = 'cpu'

from fvcore.nn import FlopCountAnalysis
import torchprofile
import json


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def count_parameters(model):
    table = PrettyTable(["Modules", "Trainable", "Parameters"])
    total_params = 0
    total_trainable_params = 0
    for name, parameter in model.named_parameters():
        # if not parameter.requires_grad:
        #     continue
        params = parameter.numel()
        table.add_row([name, parameter.requires_grad, params])
        total_params += params
        if parameter.requires_grad:
            total_trainable_params += params
    print(table)
    print(f"          Total Params: {total_params}")
    print(f"Total Trainable Params: {total_trainable_params}")
    return total_params


# Define a wrapper function for the forward method
def model_forward_wrapper(model, image1, image2, args):
    # model = IGEVStereo(args, device_ids=[0])
    return model(image1, image2, iters=args.valid_iters, test_mode=True)

# Assuming `IGEVStereo` is your model class and `args` is defined
class ModelWrapper(torch.nn.Module):
    def __init__(self, model, args):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.args = args

    def forward(self, image1, image2):
        return self.model(image1, image2, iters=self.args.valid_iters, test_mode=True)


def report_model_complexity():
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])

    model = model.module

    total_params = count_parameters(model)


    count_parameters(model)


    
    # Calculate FLOPS
    image1 = load_image(args.left_imgs)
    image2 = load_image(args.right_imgs)

    padder = InputPadder(image1.shape, divis_by=32)
    image1, image2 = padder.pad(image1, image2)

    print("fvcore FlopCountAnalysis")
    model = ModelWrapper(model, args)
    flops = FlopCountAnalysis(model, (image1, image2))
    model_statistics_json = flops.by_module()
    # print(json.dumps(model_statistics_json, indent=4))
    with open("model_statistics.json", 'w') as json_file:
        json.dump(model_statistics_json, json_file, indent=4)  # indent=4 for pretty printing

    flops = flops.total()/(10**12)
    print(f"Total TFLOPS: {flops:.4f}")

    details = {
        "parameters": total_params,
        "TFLOPS": flops,
        "restore_ckpt": os.path.basename(args.restore_ckpt),
        "mixed_precision": args.mixed_precision,
        "valid_iters": args.valid_iters,
        "hidden_dims": args.hidden_dims,
        "corr_levels": args.corr_levels,
        "corr_radius": args.corr_radius,
        "n_downsample": args.n_downsample,
        "n_gru_layers": args.n_gru_layers,
        "max_disp": args.max_disp,
    }

    with open("models_details.json", 'a') as json_file:
        json.dump(details, json_file, indent=4)
    # Calculate MACs
    # print("torchprofile.profile_macs")
    # macs = torchprofile.profile_macs(model, (image1, image2))
    # flops = 2 * macs  # Each MAC operation is typically counted as 2 FLOPS
    # print(f"Total FLOPS: {flops}")


    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/sceneflow/sceneflow.pth')
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="./demo-imgs/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="./demo-imgs/*/im1.png")

    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/Middlebury/trainingH/*/im0.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/Middlebury/trainingH/*/im1.png")
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/ETH3D/two_view_training/*/im0.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/ETH3D/two_view_training/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="./demo-output/")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    
    args = parser.parse_args()

    Path(args.output_directory).mkdir(exist_ok=True, parents=True)

    # demo(args)
    report_model_complexity()
