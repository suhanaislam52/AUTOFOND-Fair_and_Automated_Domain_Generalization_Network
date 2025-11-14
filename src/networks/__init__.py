from src.networks.base import ERM
from src.networks.fond import FOND, FOND_N, FOND_NC
from src.networks.fond_distillation import (
    FOND_Distillation_Separate_Projector, FOND_Distillation_Student_Projector,
    FOND_Distillation_Teacher_Projector)

ALGORITHMS = {
    "ERM": ERM,
    "FOND": FOND,
    "FOND_NC": FOND_NC,
    "FOND_N": FOND_N,
    "FOND_Distillation_Separate_Projector": FOND_Distillation_Separate_Projector,
    "FOND_Distillation_Teacher_Projector": FOND_Distillation_Teacher_Projector,
    "FOND_Distillation_Student_Projector": FOND_Distillation_Student_Projector,
}
