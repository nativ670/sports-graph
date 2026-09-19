from dataclasses import dataclass


@dataclass(frozen=True)
class MatchMeta:
    fps: float
    pitch_length_m: float
    pitch_width_m: float
