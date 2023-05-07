from typing import Any, Dict, Optional

import pytorch_lightning as pl
from torch.nn import Module


class FSDPStrategy(pl.strategies.FSDPStrategy):
    @property
    def model(self) -> Optional[Module]:
        """Returns the potentially wrapped LightningModule."""
        return self._model

    @model.setter
    def model(self, new_model: Optional[Module]) -> None:
        self._model = new_model

    def lightning_module_state_dict(self) -> Dict[str, Any]:
        """Returns model state."""
        if self.model is None:
            assert self.lightning_module is not None
            return self.lightning_module.state_dict()
        else:
            prefix = "_forward_module."
            state_dict = self.model.state_dict()
            state_dict = {k[len(prefix) :]: v for k, v in state_dict.items()}
            return state_dict

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        if not pl.strategies.fsdp._fsdp_available:
            return
        strategy_registry.register(
            "fsdp",
            cls,
            description="Fully Sharded Data Parallel (FSDP) training",
            override=True,
        )
        cls._registered_strategies.append("fsdp")

        strategy_registry.register(
            "fsdp_cpu_offload",
            cls,
            description="Fully Sharded Data Parallel (FSDP) training with Full Sharding and CPU Offloading",
            cpu_offload=True,
            override=True,
        )
        cls._registered_strategies.append("fsdp_cpu_offload")


def apply_all_patches():
    FSDPStrategy.register_strategies(pl.strategies.StrategyRegistry)
