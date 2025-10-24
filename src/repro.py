from __future__ import annotations
import os, sys, random, json, platform
import numpy as np


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    import os, random, numpy as np
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed); np.random.seed(seed)
    try:
        import torch as th
        th.manual_seed(seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(seed)
        if deterministic:
            try: th.use_deterministic_algorithms(True)
            except Exception: pass
            try:
                th.backends.cudnn.deterministic = True
                th.backends.cudnn.benchmark = False
            except Exception: pass
    except Exception:
        pass
    try:
        from stable_baselines3.common.utils import set_random_seed as sb3_set_seed
        sb3_set_seed(seed)
    except Exception:
        pass



def collect_versions(seed: int) -> dict:
    """
    Return a dict of versions/seeds to serialize (versions.json).
    """
    info = {
        "seed": int(seed),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    def _ver(modname, attr="__version__"):
        try:
            mod = __import__(modname)
            return getattr(mod, attr)
        except Exception:
            return None

    info.update({
        "numpy": _ver("numpy"),
        "pandas": _ver("pandas"),
        "torch": _ver("torch"),
        "torch_cuda_available": _safe_cuda(),
        "cudnn_enabled": _safe_cudnn(),
        "gymnasium": _ver("gymnasium"),
        "stable_baselines3": _ver("stable_baselines3"),
    })
    return info


def _safe_cuda():
    try:
        import torch as th
        return bool(th.cuda.is_available())
    except Exception:
        return None

def _safe_cudnn():
    try:
        import torch as th
        return bool(getattr(th.backends, "cudnn", None) and th.backends.cudnn.enabled)
    except Exception:
        return None
