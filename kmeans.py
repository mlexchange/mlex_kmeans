import os
import json
import argparse
import pathlib
import imageio
import numpy as np
from joblib import dump
from sklearn.cluster import KMeans

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

        for i, im in enumerate(images_raw):
            image = imageio.volread(im)
            images.append(image)

    images = np.array(images)
    shp = images.shape
    training_images = images.reshape((shp[0]*shp[1]*shp[2],1))

    # Get parameters
    with open(args.parameters) as f:
        parameters_f = json.load(f)
    assert parameters_f["model_name"] == "kmeans", "Incorrect model file"
    parameters = parameters_f["parameters"]
    
    # Run Kmeans
    kmeans = KMeans(n_clusters=parameters["n_clusters"],
                    init=parameters["init"],
                    n_init=parameters["n_init"],
                    max_iter=parameters["max_iter"],
                    tol=parameters["tol"],
                    random_state=parameters["random_state"],
                    algorithm=parameters["algorithm"])
    kmeans.fit(training_images)

    # Save model
    pth = os.path.join(args.model_dir, 'kmeans.joblib')
    dump(kmeans, pth)

    print('trained k-means: {}'.format(pth))
