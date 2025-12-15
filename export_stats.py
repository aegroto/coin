import lpips
import argparse
import json
import torch

from pytorch_msssim import ssim, ms_ssim

from modules.data import ImageData
from modules.device import load_device

def __load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("original_image_path", type=str)
    parser.add_argument("reconstructed_image_path", type=str)
    parser.add_argument("coin_stats_path", type=str)
    parser.add_argument("stats_dump_path", type=str)
    return parser.parse_args()


def ms_ssim_reshape(tensor):
    return tensor.unsqueeze(0)


def lpips_reshape(tensor):
    tensor = tensor.sub(0.5).mul(2.0)
    return tensor.unsqueeze(0)


def __calculate_psnr(
    original: torch.Tensor, reconstructed: torch.Tensor, value_range: float = 1.0
) -> float:
    mse = (original - reconstructed).square().mean()
    psnr = 20.0 * torch.log10(value_range / mse.sqrt())
    return psnr.item()


def main():
    args = __load_args()

    device = load_device()
    export_stats(
        args.original_image_path,
        args.reconstructed_image_path,
        args.coin_stats_path,
        args.stats_dump_path,
        device,
    )


def export_stats(
    original_image_path,
    reconstructed_image_path,
    coin_stats_path,
    stats_dump_path,
    device,
):
    original_image = ImageData(original_image_path, device)
    reconstructed_image = ImageData(reconstructed_image_path, device)

    stats = dict()
    stats["psnr"] = __calculate_psnr(original_image.tensor, reconstructed_image.tensor)

    try:
        stats["ssim"] = ssim(
            ms_ssim_reshape(original_image.tensor),
            ms_ssim_reshape(reconstructed_image.tensor),
            data_range=1.0
        ).item()
    except Exception as e:
        print(f"Cannot calculate SSIM: {e}")
        stats["ssim"] = None

    try:
        stats["ms_ssim"] = ms_ssim(
            ms_ssim_reshape(original_image.tensor),
            ms_ssim_reshape(reconstructed_image.tensor),
            data_range=1.0
        ).item()
    except Exception as e:
        print(f"Cannot calculate MS-SSIM: {e}")
        stats["ms_ssim"] = None

    try:
        loss_fn_alex = lpips.LPIPS(net="alex").to(original_image.tensor.device)
        stats["lpips"] = loss_fn_alex(
            lpips_reshape(original_image.tensor),
            lpips_reshape(reconstructed_image.tensor),
        ).item()
    except Exception as e:
        print(f"Cannot calculate LPIPS: {e}")
        stats["lpips"] = None

    coin_stats = json.load(open(coin_stats_path, "r"))
    stats["bpp"] = coin_stats["hp_bpp"][0]

    print(json.dumps(stats, indent=4))

    json.dump(stats, open(stats_dump_path, "w"), indent=4)


if __name__ == "__main__":
    main()
