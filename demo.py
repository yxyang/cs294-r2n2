'''
Demo code for the paper

Choy et al., 3D-R2N2: A Unified Approach for Single and Multi-view 3D Object
Reconstruction, ECCV 2016
'''
import os
import sys
if (sys.version_info < (3, 0)):
    raise Exception("Please follow the installation instruction on 'https://github.com/chrischoy/3D-R2N2'")

import shutil
import numpy as np
from subprocess import call

from PIL import Image
from models import load_model
from lib.config import cfg, cfg_from_list
from lib.solver import Solver
from lib.voxel import voxel2obj

DEFAULT_WEIGHTS = 'output/ResidualGRUNet/default_model/weights.npy'


def cmd_exists(cmd):
    return shutil.which(cmd) is not None


def download_model(fn):
    if not os.path.isfile(fn):
        # Download the file if doewn't exist
        print('Downloading a pretrained model')
        call(['curl', 'ftp://cs.stanford.edu/cs/cvgl/ResidualGRUNet.npy',
              '--create-dirs', '-o', fn])


def load_demo_images(image_file_name):
    ims = []
    size = (127, 127)
    for i in range(8):
        im = Image.open('custom_pics/%s/%d.jpg' % (image_file_name, i)).convert("RGB")
        im.thumbnail((127, 127))

        background = Image.new('RGB', size, (255, 255, 255))
        background.paste(
            im, (int((size[0] - im.size[0]) / 2), int((size[1] - im.size[1]) / 2))
        )
        background.save('custom_pics/%s/%d_modified.jpg' % (image_file_name, i))
        ims.append([np.array(background).transpose(
            (2, 0, 1)).astype(np.float32) / 255.])
    return np.array(ims)


def main():
    '''Main demo function'''
    # Save prediction into a file named 'prediction.obj' or the given argument
    pred_file_name = 'custom_pics/%s/%s.obj' % (sys.argv[1], sys.argv[1])

    # load images
    image_file_name = sys.argv[1]
    demo_imgs = load_demo_images(image_file_name)

    # Download and load pretrained weights
    download_model(DEFAULT_WEIGHTS)

    # Use the default network model
    NetClass = load_model('ResidualGRUNet')

    # Define a network and a solver. Solver provides a wrapper for the test function.
    net = NetClass(compute_grad=False)  # instantiate a network
    net.load(DEFAULT_WEIGHTS)                        # load downloaded weights
    solver = Solver(net)                # instantiate a solver

    # Run the network
    voxel_prediction, _ = solver.test_output(demo_imgs)

    # Save the prediction to an OBJ file (mesh file).
    voxel2obj(pred_file_name, voxel_prediction[0, :, 1, :, :] > cfg.TEST.VOXEL_THRESH)

    # Use meshlab or other mesh viewers to visualize the prediction.
    # For Ubuntu>=14.04, you can install meshlab using
    # `sudo apt-get install meshlab`
    if cmd_exists('meshlab'):
        call(['meshlab', pred_file_name])
    else:
        print('Meshlab not found: please use visualization of your choice to view %s' %
              pred_file_name)


if __name__ == '__main__':
    # Set the batch size to 1
    cfg_from_list(['CONST.BATCH_SIZE', 1])
    main()
