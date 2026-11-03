"""Change the GMM-MIGT preprocessing parameters and run ablations.

Every parameter has a default (thesis values) and can be changed by the user:
number of GMM clusters, EM tolerance / iterations / covariance / initialisation,
PCA cumulative percent variance, and each step can be switched off.

    python examples/04_custom_preprocessing.py [path/to/dataset]
"""

from _demo_data import dataset_from_argv

from gmm_migt import GMMMIGTSplitter

dataset = dataset_from_argv()

configs = {
    "GMM-MIGT default (K=4, CPV 95%)": dict(),
    "K=6 clusters": dict(n_components=6),
    "K=2 clusters, CPV 90%": dict(n_components=2, pca_variance=0.90),
    "diag covariance, random init": dict(gmm_covariance_type="diag", gmm_init="random"),
    "tighter EM": dict(gmm_tol=1e-4, gmm_max_iter=300),
    "fit GMM on 2000 pixels/image": dict(gmm_max_pixels=2000),
    "EM-GMM only (no PCA)": dict(use_pca=False),
    "PCA only (no EM-GMM)": dict(use_gmm=False),
    "no preprocessing (= MIGT)": dict(use_gmm=False, use_pca=False),
}

for name, kw in configs.items():
    res = GMMMIGTSplitter(dataset, verbose=False, **kw).plan()
    sizes = {c: [b.size for b in r.bins] for c, r in res.classes.items()}
    print(f"{name:<34} bins per class: {sizes}")
