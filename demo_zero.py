from itertools import chain
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import timm
import numpy as np
from PIL import ImageDraw

import models_mae_cross
from util.misc import measure_time


assert "0.4.5" <= timm.__version__ <= "0.4.9"  # version check
device = torch.device('cuda')
shot_num = 0


def load_image(img_path: str):
    image = Image.open(img_path).convert('RGB')
    image.load()
    W, H = image.size

    # Resize the image size so that the height is 384
    new_H = 384
    new_W = 16 * int((W / H * 384) / 16)
    image = transforms.Resize((new_H, new_W))(image)
    Normalize = transforms.Compose([transforms.ToTensor()])
    image = Normalize(image)

    # Coordinates of the exemplar bound boxes
    # Not needed for zero-shot counting
    boxes = torch.Tensor([])
    return image, boxes, W, H


def run_one_image(samples, boxes, model, output_path, img_name, old_w, old_h, img_path, threshold=0.1, filter_size=15):
    _, _, h, w = samples.shape

    density_map = torch.zeros([h, w])
    density_map = density_map.to(device, non_blocking=True)
    start = 0
    prev = -1
    with measure_time() as et:
        with torch.no_grad():
            while start + 383 < w:
                output, = model(samples[:, :, :, start:start + 384], boxes, shot_num)
                output = output.squeeze(0)
                b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
                d1 = b1(output[:, 0:prev - start + 1])
                b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
                d2 = b2(output[:, prev - start + 1:384])

                b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
                density_map_l = b3(density_map[:, 0:start])
                density_map_m = b1(density_map[:, start:prev + 1])
                b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
                density_map_r = b4(density_map[:, prev + 1:w])

                density_map = density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2
                #density_map = (density_map_l + density_map_r + density_map_m + d1 + d2) / 3

                prev = start + 383
                start = start + 128
                if start + 383 >= w:
                    if start == w - 384 + 128:
                        break
                    else:
                        start = w - 384

        pred_cnt = torch.sum(density_map / 64).item()

    # Visualize the prediction
    # Use the original (preprocessed, but not normalized) image for visualization
    orig_img = Image.open(img_path).convert('RGB')
    orig_img = orig_img.resize((w, h), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(orig_img)
    # Mark exactly N highest density points, where N is the predicted count
    # Find all local maxima in the density map
    import scipy.ndimage
    np_density = density_map.cpu().numpy()
    local_max = scipy.ndimage.maximum_filter(np_density, size=filter_size) == np_density
    maxima_coords = np.argwhere(local_max & (np_density > threshold))
    maxima_values = np_density[local_max & (np_density > threshold)]
    if len(maxima_coords) > 0:
        sorted_indices = np.argsort(maxima_values)[::-1]
        top_n = min(int(round(pred_cnt)), len(sorted_indices))
        selected_coords = maxima_coords[sorted_indices[:top_n]]
        for y, x in selected_coords:
            draw.ellipse([(x-1, y-1), (x+1, y+1)], fill=(0,255,0))  # Smaller green dot
    # Advanced peak detection using skimage.feature.peak_local_max
    from skimage.feature import peak_local_max
    # Normalize density map for peak detection
    norm_density = (np_density - np_density.min()) / (np_density.max() - np_density.min() + 1e-8)
    # Use peak_local_max with min_distance and threshold_abs
    coordinates = peak_local_max(norm_density, min_distance=filter_size//2, threshold_abs=threshold, num_peaks=int(round(pred_cnt)))
    print(f"[INFO] peak_local_max found {len(coordinates)} peaks (target: {int(round(pred_cnt))})")
    for y, x in coordinates:
        draw.ellipse([(x-1, y-1), (x+1, y+1)], fill=(0,255,0))  # Small green dot
    draw.text((w-70, h-50), f"{pred_cnt:.3f}", (255, 255, 255))
    orig_img = orig_img.resize((old_w, old_h), Image.Resampling.BILINEAR)
    orig_img.save(output_path / f'viz_{img_name}.jpg')
    return pred_cnt, et


if __name__ == '__main__':
    # get parameters
    p = ArgumentParser()
    p.add_argument("--input_path", type=Path, required=True)
    p.add_argument("--output_path", type=Path, default="results")
    p.add_argument("--model_path", type=Path, default=r"weights/FSC147.pth")
    p.add_argument("--peak_threshold", type=float, default=0.0, help="Threshold for local maxima in density map (0-1, as fraction of max)")
    p.add_argument("--filter_size", type=int, default=15, help="Size of maximum filter for local maxima detection")
    args = p.parse_args()

    args.output_path.mkdir(exist_ok=True, parents=True)

    # Prepare model
    model = models_mae_cross.__dict__['mae_vit_base_patch16'](norm_pix_loss='store_true')
    model.to(device)
    model_without_ddp = model

    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    print(f"Resume checkpoint {args.model_path}")

    model.eval()

    # Test on the new image
    if args.input_path.is_dir():
        inputs = sorted(list(chain(args.input_path.glob("*.jpg"), args.input_path.glob("*.png"))))
        for i, img_path in enumerate(inputs):
            samples, boxes, old_w, old_h = load_image(img_path)
            samples = samples.unsqueeze(0).to(device, non_blocking=True)
            boxes = boxes.unsqueeze(0).to(device, non_blocking=True)
            result, elapsed_time = run_one_image(samples, boxes, model, args.output_path, img_path.stem, old_w, old_h, img_path, threshold=args.peak_threshold, filter_size=args.filter_size)
            print(f"[{i+1: 3}/{len(inputs)}] {img_path.name}:\tcount = {result:5.2f}  -  time = {elapsed_time.duration:5.2f}")
    else:
        samples, boxes, old_w, old_h = load_image(args.input_path)
        samples = samples.unsqueeze(0).to(device, non_blocking=True)
        boxes = boxes.unsqueeze(0).to(device, non_blocking=True)
        result, elapsed_time = run_one_image(samples, boxes, model, args.output_path, args.input_path.stem, old_w, old_h, args.input_path, threshold=args.peak_threshold, filter_size=args.filter_size)
        print("Count:", result, "- Time:", elapsed_time.duration)