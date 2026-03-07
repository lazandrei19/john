from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from romanian_whist.rules.config import OneCardMode


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    player_counts: Tuple[int, ...]
    one_card_modes: Tuple[OneCardMode, ...]


class CurriculumScheduler:
    def __init__(self, total_updates: int):
        self.total_updates = max(1, total_updates)
        self.stages = (
            CurriculumStage("stage_1", (4,), (OneCardMode.REGULAR,)),
            CurriculumStage("stage_2", (3, 4, 5, 6), (OneCardMode.REGULAR,)),
            CurriculumStage("stage_3", (3, 4, 5, 6), (OneCardMode.REGULAR, OneCardMode.FOREHEAD, OneCardMode.BLIND)),
        )

    def stage_for_update(self, update_index: int) -> CurriculumStage:
        progress = float(update_index + 1) / float(self.total_updates)
        if progress <= 0.2:
            return self.stages[0]
        if progress <= 0.6:
            return self.stages[1]
        return self.stages[2]
