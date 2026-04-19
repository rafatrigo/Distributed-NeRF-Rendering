import os, sys
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from nerf_core.model import init_model, get_rays, render_rays, pose_spherical

#TODO: Add parameters to run on terminal 
"""
Parametes to:
- npz file path
- num of samples 
- num of iterrations
- only run the tests 
- show images during training 
- generate video
"""
if __name__ == "__main__":

    data = np.load('data/tiny_nerf_data.npz')
    images = data['images']
    poses = data['poses']
    focal = data['focal']
    H, W = images.shape[1:3]
    print(images.shape, poses.shape, focal)

    testimg, testpose = images[101], poses[101]
    images = images[:100,...,:3]
    poses = poses[:100]

    plt.imshow(testimg)
    plt.show()

    model = init_model()
    optimizer = tf.keras.optimizers.Adam(5e-4)

    N_samples = 64
    N_iters = 500
    psnrs = []
    iternums = []
    i_plot = 100

    import time
    t = time.time()
    for i in range(N_iters+1):
        
        img_i = np.random.randint(images.shape[0])
        target = images[img_i]
        pose = poses[img_i]
        rays_o, rays_d = get_rays(H, W, focal, pose)
        with tf.GradientTape() as tape:
            rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples, rand=True)
            loss = tf.reduce_mean(tf.square(rgb - target))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
       
        print(f"Iteration {i} concluded!")

        if i%i_plot==0:
            print(i, (time.time() - t) / i_plot, 'secs per iter')
            t = time.time()
            
            # Render the holdout view for logging
            rays_o, rays_d = get_rays(H, W, focal, testpose)
            rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
            loss = tf.reduce_mean(tf.square(rgb - testimg))
            psnr = -10. * tf.math.log(loss) / tf.math.log(10.)

            psnrs.append(psnr.numpy())
            iternums.append(i)
            
            plt.figure(figsize=(10,4))
            plt.subplot(121)
            plt.imshow(rgb)
            plt.title(f'Iteration: {i}')
            plt.subplot(122)
            plt.plot(iternums, psnrs)
            plt.title('PSNR')
            plt.show()

    print('Done')
    
    # save model weights
    model.save_weights('weights.h5')
    print("weights saved with success!")

## video 360

    frames = []
    for th in tqdm(np.linspace(0., 360., 120, endpoint=False)):
        c2w = pose_spherical(th, -30., 4.)
        rays_o, rays_d = get_rays(H, W, focal, c2w[:3,:4])
        rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
        frames.append((255*np.clip(rgb,0,1)).astype(np.uint8))

    import imageio
    f = 'video.mp4'
    imageio.mimwrite(f, frames, fps=30, quality=7)


