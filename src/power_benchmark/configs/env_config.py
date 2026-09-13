from typing import Any, Optional, Dict

from pydantic import BaseModel


class EnvConfig(BaseModel):
    reward: Optional[Dict[str, float]] = None
    worst_reward: float = -1000
    nminus1: bool = False
    nminus1_topk: float = 100.0
    verify_actions: bool = True

    def model_post_init(self, context: Any, /) -> None:
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reward": self.reward,
            "worst_reward": self.worst_reward,
            "nminus1": self.nminus1,
            "nminus1_topk": self.nminus1_topk,
            "verify_actions": self.verify_actions,
        }