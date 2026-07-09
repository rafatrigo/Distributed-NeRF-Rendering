import imageio
import time
import numpy as np
import csv
import os
from tqdm import tqdm

from nerf_core.model import init_model, pose_spherical, render_rays, get_rays

#TODO: Create parameters to run on terminal
"""
Parameters:
    - weights file path
    - H, W and focal variables
    - image or video
    - output filename
"""
if __name__ == "__main__":
    model = init_model()

    print("Initing model...")

    csv_file = "single_machine_metrics.csv"


    loop_count = 0
    while(loop_count < 10):
        start = time.perf_counter()

        model.load_weights('data/weights.h5')

        print("Loading model...")

        H, W = 100, 100 # Image resolution
        focal = 138.88
        frames = []
        print("Creating video...")
        for th in tqdm(np.linspace(0., 360., 120, endpoint=False)):
            c2w = pose_spherical(th, -30., 4.)
            rays_o, rays_d = get_rays(H, W, focal, c2w[:3,:4])
            rgb_map, depth_map, acc_map= render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=64)
            frames.append((255*np.clip(rgb_map,0,1)).astype(np.uint8))
        
        end = time.perf_counter()

        total_time = end - start

        # verify if file exist
        arquivo_existe = os.path.isfile(csv_file)

        with open(csv_file, "a", newline="") as file:
            writer = csv.writer(file)

            # write the header only in the first time
            if not arquivo_existe:
                writer.writerow(["num_workers","image_width","image_height","execution_time_ms"])

            # write data
            writer.writerow([1,W, H, total_time*1000])

    f = 'video.mp4'
    imageio.mimwrite(f, frames, fps=30, quality=7)


