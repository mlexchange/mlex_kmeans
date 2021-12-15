import os
import numpy as np
import pathlib
import argparse
import imageio
from joblib import load
import json

from model_validation import TestingParameters


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_stack', help='image filepath')
    parser.add_argument('model_file', help='model filepath')
    parser.add_argument('output_dir', help='directory for outputs')
    parser.add_argument('parameters', help='dictionary that contains testing parameters')
    
    args = parser.parse_args()
    # Load testing parameters
    parameters = TestingParameters(**json.loads(args.parameters))
    
    image_stack = pathlib.Path(args.image_stack)
    model_file = pathlib.Path(args.model_file)
    output_dir = pathlib.Path(args.output_dir)

    images = imageio.volread(image_stack)
    shp = images.shape
    plchldr = 1
    for i in range(len(shp)-1):
        plchldr = shp[i] * plchldr
    test_images = images.reshape(plchldr,shp[len(shp)-1])

    kmeans = load(model_file)
    outputs = kmeans.predict(test_images).reshape(shp[0], shp[1], shp[2])
    io_path = pathlib.Path(args.output_dir)
    io_path.mkdir(parents=True, exist_ok=True)
    
    for index, out in enumerate(outputs):
        if index % parameters.show_progress == 0:
            output_f_name = output_dir / '{}-.dat'.format(index)
            np.savetxt(str(output_f_name), out)
            imageio.imsave(str(output_dir / '{}-classified.tif'.format(index)), out)
            print('classified\t{}'.format(index), flush=True)
