import os
import json
import argparse
import pathlib
import imageio
import numpy as np
from joblib import dump
from sklearn.cluster import KMeans

from model_validation import TrainingParameters

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_stack', help='image filepath')
    parser.add_argument('model_dir', help='path to model (output) directory')
    parser.add_argument('parameters', help='dictionary that contains training parameters')
    args = parser.parse_args()
    
    # Get images
    if args.image_stack[-4:] == '.tif':
        images = imageio.volread(args.image_stack)
    else:
        images = []
        images_path = pathlib.Path(args.image_stack)
        images_raw = images_path.glob('*.tif')

        for im in images_raw:
            image = imageio.volread(im)
            images.append(image)

    images = np.array(images)
    shp = images.shape
    plchldr = 1
    print(f'shape {shp}')
    for i in range(len(shp)-1):
        plchldr = shp[i] * plchldr
    training_images = images.reshape(plchldr,shp[len(shp)-1])

        # Load training parameters
    if args.parameters is not None:
        parameters = TrainingParameters(**json.loads(args.parameters))
        
    # Run Kmeans
    kmeans = KMeans(n_clusters=parameters.n_clusters,
                    max_iter=parameters.max_iter
                    )
    kmeans.fit(training_images)

    # Save model
    io_path = pathlib.Path(args.model_dir)
    io_path.mkdir(parents=True, exist_ok=True)
    pth = os.path.join(args.model_dir, 'kmeans.joblib')
    dump(kmeans, pth)

    print('trained k-means: {}'.format(pth))
