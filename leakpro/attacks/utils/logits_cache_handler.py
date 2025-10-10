from datetime import datetime
import os
import tempfile
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
import json

class LogitsCacheHandler():
    """
    Handles saving, loading, and validating cached logits and metadata
    for different attack configurations.
    """

    def __init__(self, base_dir: str = "./leakpro_output"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _paths(self, config_hash: str) -> tuple[Path, Path]:
        logits = self.base_dir / f"{config_hash}_logits.npz"
        meta = self.base_dir / f"{config_hash}_meta.npz"
        return logits, meta

    @staticmethod
    def _audit_indices_hash(audit_indices: Any) -> str:
        """Stable SHA256 hash for identifying audit index sets."""
        b = json.dumps(list(map(int,audit_indices)), sort_keys=True).encode("utf-8")
        return hashlib.sha256(b).hexdigest()

    def exists(self, config_hash) -> bool:
        logits, meta = self._paths(config_hash)
        return logits.exists() and meta.exists()
    
    def save(self,
             config_hash: str,
             shadow_models_logits: np.ndarray,
             in_indices_masks: np.ndarray,
             audit_data_indices: np.ndarray,
             in_members: np.ndarray,
             out_members: np.ndarray,
             extra_metadata: Optional[Dict[str, Any]] = None) -> None:

        """Save computed logits and metadata atomically."""
        logits_path, meta_path = self._paths(config_hash)
        
        # Prep data
        arrays = {
            "shadow_models_logits": np.array(shadow_models_logits),
            "in_indices_masks": np.array(in_indices_masks),
            "audit_data_indices": np.array(audit_data_indices),
            "in_members": np.array(in_members),
            "out_members": np.array(out_members)
        }
        
        # Compute checksum
        hasher = hashlib.sha256()
        for name in sorted(arrays.keys()):
            hasher.update(name.encode("utf-8"))
            hasher.update(np.ascontiguousarray(arrays[name]).tobytes())
        arrays_checksum = hasher.hexdigest()

        metadata = {
            "config_hash": config_hash,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "n_shadowmodels": int(arrays["shadow_models_logits"].shape[1]),
            "audit_indices_hash": self._audit_indices_hash(audit_data_indices),
            "arrays_checksum": arrays_checksum
        }

        if extra_metadata:
            metadata.update(extra_metadata)

        # Atomic save to avoid cooruption
        tmp_npz = tempfile.NamedTemporaryFile(delete=False, suffix=".npz")
        np.savez_compressed(tmp_npz, **arrays)
        os.replace(tmp_npz, logits_path)

        with open(meta_path, "W") as f:
            json.dump(metadata, f, indent=2)


