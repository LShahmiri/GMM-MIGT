"""Use the EM-GMM + PCA preprocessing and MI functions on individual images.

    python examples/06_single_image_preprocessing.py [image1] [image2]
"""

import os
import sys
import tempfile

import numpy as np

from gmm_migt import (
    PreprocessingConfig,
    compute_mi,
    em_gmm_colour_reduction,
    load_image,
    pca_reconstruction,
    preprocess_image,
    save_image,
)

if len(sys.argv) > 2:
    a, b = load_image(sys.argv[1]), load_image(sys.argv[2])
else:
    from sklearn.datasets import load_sample_images

    imgs = load_sample_images().images
    a, b = imgs[0][:, :, ::-1].copy(), imgs[1][:, :, ::-1].copy()  # RGB -> OpenCV BGR

# Step 1: EM-GMM colour reduction (K = 4)
reduced, gmm_info = em_gmm_colour_reduction(a, n_components=4, tol=1e-3, max_iter=100, random_state=0)
print("EM converged:", gmm_info.converged, "after", gmm_info.n_iter, "iterations")
print("mixture weights:", np.round(gmm_info.weights, 3))
print("distinct colours:", len(np.unique(reduced.reshape(-1, reduced.shape[-1]), axis=0)))

# Step 2: per-channel PCA keeping 95% cumulative variance
recon, pca_info = pca_reconstruction(reduced, variance=0.95)
print("principal components per channel:", pca_info.n_components, "CPV:", np.round(pca_info.cpv, 4))

# Both steps at once, with custom parameters
cfg = PreprocessingConfig(n_components=4, pca_variance=0.95, random_state=0)
pa, _ = preprocess_image(a, cfg)
pb, _ = preprocess_image(b, cfg)

out = os.path.join(tempfile.mkdtemp(), "processed.png")
save_image(out, pa)
print("processed image written to", out)

print("MI original  :", compute_mi(a, b, mode="color"))
print("MI processed :", compute_mi(pa, pb, mode="color"))
