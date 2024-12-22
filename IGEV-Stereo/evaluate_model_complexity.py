import sys
sys.path.append('core')
import argparse
import os

from pathlib import Path

from igev_stereo import IGEVStereo
from prettytable import PrettyTable
from utils.utils import InputPadder
import logging

import numpy as np
from PIL import Image
import torch
# DEVICE = 'cpu'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

from fvcore.nn import FlopCountAnalysis
# import torchprofile
import json

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v
    return new_state_dict

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    # if has 3 channels, discard the fourth
    if img.shape[2] == 4:
        img = img[:,:,:3]
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

@torch.no_grad()
def report_model_complexity():
    # args.n_downsample = 2
    # args.hidden_dims = [128]*3
    # args.corr_levels = 2
    # args.corr_radius = 4
    # args.n_gru_layers = 3
    # args.max_disp = 192
    # args.restore_ckpt = './pretrained_models/sceneflow/sceneflow.pth'
    # args.valid_iters = 32
    # args.mixed_precision = True


    # model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model = IGEVStereo(args)

    state_dict = torch.load(args.restore_ckpt, map_location=torch.device(DEVICE))

    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)

    # model.load_state_dict(torch.load(args.restore_ckpt, map_location=torch.device(DEVICE)))
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
    # if args.restore_ckpt is not None:
    #     assert args.restore_ckpt.endswith(".pth")
    #     logging.info("Loading checkpoint...")
    #     checkpoint = torch.load(args.restore_ckpt)
    #     model.load_state_dict(checkpoint, strict=True)
    #     logging.info(f"Done loading checkpoint")

    # model = model.module
    model.to(DEVICE)
    model.eval()

    total_params = count_parameters(model)
    print(f"The model has {format(total_params/1e6, '.2f')}M learnable parameters.")



    ######################
    # My images
    # Calculate FLOPS
    image1 = load_image(args.left_imgs)
    image2 = load_image(args.right_imgs)
    ##############################


    padder = InputPadder(image1.shape, divis_by=32)
    image1, image2 = padder.pad(image1, image2)

    with torch.amp.autocast('cuda', enabled=args.mixed_precision):
        disp = model(image1, image2, iters=args.valid_iters, test_mode=True)

    from torchviz import make_dot
    dot = make_dot(disp, params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render('model_graph')



    print(torch.cuda.memory_summary())
    
    print("fvcore FlopCountAnalysis")
    model = ModelWrapper(model, args)

    # output = model(image1, image2)
    # from torchviz import make_dot
    # dot = make_dot(output, params=dict(model.named_parameters()))
    # dot.format = 'png'
    # dot.render('model_graph')

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
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
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
