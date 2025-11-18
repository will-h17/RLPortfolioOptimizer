from stable_baselines3.common.logger import configure

# Make tensorboard optional (not required for core functionality)
try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False
    SummaryWriter = None  # type: ignore

def make_loggers(log_dir: str):
    # SB3 logger (tensorboard output is optional)
    log_outputs = ["stdout", "csv"]
    if _TENSORBOARD_AVAILABLE:
        log_outputs.append("tensorboard")
    
    sb3_logger = configure(log_dir, log_outputs)
    
    # for ad-hoc logging outside SB3 (optional)
    if _TENSORBOARD_AVAILABLE:
        tb = SummaryWriter(log_dir)
    else:
        tb = None  # Return None if tensorboard not available
    
    return sb3_logger, tb