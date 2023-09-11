import argparse
from pytorch_msssim import ms_ssim, ssim
import imageio
import json
import os
import torch
import util
from siren import Siren
from torchvision import transforms
from torchvision.utils import save_image

def psnr(img1, img2, range=255.0):
    mse = torch.mean((img1 - img2) ** 2)
    psnr = 20 * torch.log10(range / torch.sqrt(mse))

    return psnr.item()

def ms_ssim_reshape(tensor):
    return tensor.unsqueeze(0)

parser = argparse.ArgumentParser()
parser.add_argument("image_id")
parser.add_argument("config_id")
parser.add_argument("output_folder")
parser.add_argument("-fp", action="store_true")

args = parser.parse_args()

# Set up torch and cuda
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

# Load image
img = imageio.v2.imread(f"kodak-dataset/kodim{str(args.image_id).zfill(2)}.png")
img = transforms.ToTensor()(img).float().to(device, dtype)

confs = args.config_id.split("_")
num_layers = int(confs[0])
layer_size = int(confs[1])

# Setup model
func_rep = Siren(
    dim_in=2,
    dim_hidden=layer_size,
    dim_out=3,
    num_layers=num_layers,
    final_activation=torch.nn.Identity()
).to(device)

model_path = f"logs_dir/{args.config_id}/best_model_{args.image_id}.pt"
state_dict = torch.load(open(model_path, "rb"))
func_rep.load_state_dict(state_dict)

# Set up training
coordinates, original = util.to_coordinates_and_features(img)
coordinates, original = coordinates.to(device, dtype), original.to(device, dtype)

# Calculate model size. Divide by 8000 to go from bits to kB
model_size = util.model_size_in_bits(func_rep) / 8000.
fp_bpp = util.bpp(model=func_rep, image=img)

stats = dict()

# Convert model and coordinates to half precision. Note that half precision
# torch.sin is only implemented on GPU, so must use cuda
if torch.cuda.is_available():
    if args.fp:
        func_rep = func_rep.to('cuda')
        coordinates = coordinates.to('cuda')
    else:
        func_rep = func_rep.half().to('cuda')
        coordinates = coordinates.half().to('cuda')

    # Calculate model size in half precision
    hp_bpp = util.bpp(model=func_rep, image=img)
    stats['bpp'] = hp_bpp
    stats['state_bpp'] = hp_bpp

    # Compute image reconstruction and PSNR
    with torch.no_grad():
        img_recon = func_rep(coordinates)
        original = original.reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1).float()
        img_recon = img_recon.reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1).float()
        save_image(torch.clamp(img_recon, 0, 1).to('cpu'), args.output_folder + f'/decoded.png')
        original = original.mul(255.0)
        img_recon = img_recon.mul(255.0)
        stats['psnr'] = psnr(original, img_recon)

        try:
            ssim_value = ssim(ms_ssim_reshape(original), ms_ssim_reshape(img_recon)).item()
        except Exception as e:
            print(f"Cannot calculate SSIM: {e}")
            ssim_value = None

        try:
            ms_ssim_value = ms_ssim(ms_ssim_reshape(original), ms_ssim_reshape(img_recon)).item()
        except Exception as e:
            print(f"Cannot calculate MS-SSIM: {e}")
            ms_ssim_value = None

        stats["ssim"] = ssim_value
        stats["ms-ssim"] = ms_ssim_value

print(stats)
json.dump(stats, open(f"{args.output_folder}/stats.json", "w"))
