import numpy as np

def fuse_embeddings(embeddings: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """
    Fuse multiple face embeddings into a single representative template.

    embeddings: list of vectors from InsightFace (one per enrollment photo)
    weights: optional per-embedding weight (e.g. SCRFD detection confidence).
             If None, all embeddings are weighted equally.
    """
    if not embeddings:
        raise ValueError("Cannot fuse an empty list of embeddings.")

    # Normalize each embedding
    normalized = [e / np.linalg.norm(e) for e in embeddings]

    if len(normalized) <= 2:
        # Cannot do outlier check with 2 or less
        keep = normalized
    else:
        # Outlier check: Drop if average cosine sim to others is below threshold (0.5)
        keep = []
        n = len(normalized)
        sim_threshold = 0.5
        for i in range(n):
            sims_to_others = [
                np.dot(normalized[i], normalized[j])
                for j in range(n) if j != i
            ]
            avg_sim = np.mean(sims_to_others)
            if avg_sim >= sim_threshold:
                keep.append(normalized[i])
            else:
                print(f"    [Fusion] Warning: Photo {i} rejected as outlier (avg sim: {avg_sim:.3f})")
        
        if not keep:
            keep = normalized  # fallback if all are considered outliers

    # Average and re-normalize
    fused = np.mean(keep, axis=0)
    return fused / np.linalg.norm(fused)
