import imageio
import numpy as np
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

    model.load_weights('data/weights.h5')

    print("Loading model...")

    H, W = 100, 100 # Image resolution
    focal = 138.88

    # print("Creating video...")
    #
    # for th in tqdm(np.linspace(0., 360., 120, endpoint=False)):
    #     c2w = pose_spherical(th, -30., 4.)
    #     rays_o, rays_d = get_rays(H, W, focal, c2w[:3,:4])
    #     rgb_map, depth_map, acc_map= render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=64)
    #     frames.append((255*np.clip(rgb_map,0,1)).astype(np.uint8))
    #
    # f = 'video.mp4'
    # imageio.mimwrite(f, frames, fps=30, quality=7)

    print("Creating image...")

    #camera position
    horizontal_angle = 45.0
    c2w = pose_spherical(horizontal_angle, -30., 4.)

    rays_o, rays_d = get_rays(H, W, focal, c2w[:3,:4])
    rgb_map, depth_map, acc_map = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=64,)
    
    img_final = np.clip(rgb_map.numpy(), 0, 1)

    # convert float [0, 1] to integer [0, 255]
    img_uint8 = (img_final*255).astype(np.uint8)

    # save image
    imageio.imwrite('data/img_final.png', img_uint8)

