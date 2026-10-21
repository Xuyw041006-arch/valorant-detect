"""A learned, explicitly limited missing-Agent rejection adapter.

The binary adapter is trained on embeddings from supported identities and
held-out identity categories (Fade/Killjoy). It is not a guarantee of detecting
all unseen Agents. Classifier weight SHA is bound in the adapter file.
"""

import json, math
from pathlib import Path


class IdentityRejector:
    def __init__(self, path, weights_sha256=None):
        self.metadata = json.loads(Path(path).read_text())
        if weights_sha256 and self.metadata["classifier_sha256"] != weights_sha256:
            raise ValueError(
                "Identity rejection adapter uses different classifier weights"
            )
        self.threshold = float(self.metadata["threshold"])
        self.coef = self.metadata["coef"]
        self.intercept = float(self.metadata["intercept"])

    def accept(self, feature):
        import numpy as np

        x = np.asarray(feature, dtype=float)
        x /= np.linalg.norm(x) + 1e-12
        if len(x) != len(self.coef):
            raise ValueError("Rejection feature dimension mismatch")
        z = float(x @ np.asarray(self.coef) + self.intercept)
        prob = 1 / (1 + math.exp(-max(-50, min(50, z))))
        return prob < self.threshold, prob


def predict_with_embeddings(model, images, **kwargs):
    """Capture the classifier's penultimate linear-layer inputs for this batch."""
    import numpy as np

    if not images:
        return [], np.empty((0, 0))
    head = model.model.model[-1]
    if not hasattr(head, "linear"):
        raise ValueError("Expected pinned YOLO classification linear head")
    features = []
    handle = head.linear.register_forward_pre_hook(
        lambda module, args: features.append(args[0].detach().float().cpu().numpy())
    )
    try:
        results = model.predict(images, **kwargs)
    finally:
        handle.remove()
    if not features:
        raise RuntimeError("Classifier embedding hook returned no features")
    # Ultralytics may run a warmup batch before the actual user batch.
    embeddings = np.concatenate(features, axis=0)[-len(images) :]
    if len(embeddings) != len(images):
        raise RuntimeError("Embedding count differs from input crops")
    return results, embeddings
