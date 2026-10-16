from .contrastive import ContrastiveTrainer
from .jepa import JEPATrainer
from .mae import MAETrainer
from .metrics import MaskedMetrics
from .pretrainer import Pretrainer
from .trainer import Trainer

# Pretraining methods by the `name` of the `pretrain` config group.
PRETRAINERS = {"jepa": JEPATrainer, "mae": MAETrainer, "contrastive": ContrastiveTrainer}

__all__ = [
    "ContrastiveTrainer",
    "JEPATrainer",
    "MAETrainer",
    "MaskedMetrics",
    "PRETRAINERS",
    "Pretrainer",
    "Trainer",
]
