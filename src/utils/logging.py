from stable_baselines3.common.logger import configure
from torch.utils.tensorboard import SummaryWriter

def make_loggers(log_dir: str):
    # SB3 logger
    sb3_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    # for ad-hoc logging outside SB3
    tb = SummaryWriter(log_dir)
    return sb3_logger, tb