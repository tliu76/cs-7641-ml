"""
Extra credit -- one nonlinear manifold learning method.

Default is t-SNE (configurable via UL_MANIFOLD: tsne|isomap|lle|spectral, or
umap if umap-learn is installed). The goal is to check whether a NONLINEAR
embedding reveals Cover_Type structure that the required LINEAR methods (PCA in
particular) do not surface in 2D.

Scope / leakage note: t-SNE is a transductive visualization method with no clean
out-of-sample transform, so it is used ONLY for visualization and qualitative
comparison here, never as downstream NN features -- this keeps the train/test
protocol honest (FAQ §11). We embed a fixed subsample for legibility and compare
the 2D t-SNE map against the top-2 PCA map on the SAME points.
"""

from __future__ import annotations

import time

import numpy as np

from . import config as C
from .seeds import set_all_seeds


def _make_manifold(method: str, seed: int):
    m = method.lower()
    if m == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=C.MANIFOLD_COMPONENTS, init="pca",
                    perplexity=30, learning_rate="auto", random_state=seed)
    if m == "isomap":
        from sklearn.manifold import Isomap
        return Isomap(n_components=C.MANIFOLD_COMPONENTS, n_neighbors=15)
    if m == "lle":
        from sklearn.manifold import LocallyLinearEmbedding
        return LocallyLinearEmbedding(n_components=C.MANIFOLD_COMPONENTS,
                                      n_neighbors=15, random_state=seed)
    if m == "spectral":
        from sklearn.manifold import SpectralEmbedding
        return SpectralEmbedding(n_components=C.MANIFOLD_COMPONENTS,
                                 n_neighbors=15, random_state=seed)
    if m == "umap":
        import umap  # optional dependency
        return umap.UMAP(n_components=C.MANIFOLD_COMPONENTS, random_state=seed)
    raise ValueError(f"Unknown manifold method: {method}")


def embedding_diagnostics(X: np.ndarray, emb: np.ndarray, y: np.ndarray,
                          seed: int = 0, k: int = 10) -> dict:
    """Quantify what a 2-D embedding preserves, so the extra credit is not eyeballed.

    Three post-hoc measures, none of which touches ``y`` while fitting:

      trustworthiness -- label-free. Fraction of each point's k nearest
          neighbours in the embedding that were also near it in the original
          54-d space. This is the direct test of "did the map preserve local
          structure", independent of Cover_Type.
      knn label agreement -- share of the k nearest neighbours IN THE EMBEDDING
          carrying the same Cover_Type. High values mean local neighbourhoods are
          label-coherent even if the global layout is not.
      K-Means ARI/NMI -- cluster the embedding (label-free), score afterwards.
          Directly comparable to the linear-DR rows of Part 3.
    """
    from sklearn.cluster import KMeans
    from sklearn.manifold import trustworthiness
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.neighbors import NearestNeighbors

    tw = float(trustworthiness(X, emb, n_neighbors=k))

    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    idx = nn.kneighbors(emb, return_distance=False)[:, 1:]     # drop self
    knn_agree = float(np.mean(y[idx] == y[:, None]))

    km = KMeans(n_clusters=C.N_CLASSES, n_init=10, random_state=seed)
    lab = km.fit_predict(emb)                                  # fit without labels
    return {
        "trustworthiness": tw,
        "knn_label_agreement": knn_agree,
        "kmeans_ari": float(adjusted_rand_score(y, lab)),
        "kmeans_nmi": float(normalized_mutual_info_score(y, lab)),
    }


def run_manifold(X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    """Embed a subsample with the chosen nonlinear method + top-2 PCA baseline."""
    set_all_seeds(seed)
    n = min(C.MANIFOLD_SAMPLE, len(X))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    Xs, ys = X[idx], y[idx]

    from sklearn.decomposition import PCA
    pca2 = PCA(n_components=2, random_state=seed).fit_transform(Xs)

    t0 = time.perf_counter()
    emb = _make_manifold(C.MANIFOLD_METHOD, seed).fit_transform(Xs)
    runtime = time.perf_counter() - t0

    return {
        "method": C.MANIFOLD_METHOD,
        "embedding": emb,
        "pca2": pca2,
        "labels": ys,
        "n": n,
        "runtime_s": runtime,
    }
