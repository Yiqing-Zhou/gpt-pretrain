from typing import Any, Dict, Optional, Union

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


class DeepSpeedStrategy(pl.strategies.DeepSpeedStrategy):
    def _create_default_config(
        self,
        zero_optimization: bool,
        zero_allow_untested_optimizer: bool,
        logging_batch_size_per_gpu: Union[str, int],
        partition_activations: bool,
        cpu_checkpointing: bool,
        contiguous_memory_optimization: bool,
        synchronize_checkpoint_boundary: bool,
        offload_optimizer: bool,
        offload_parameters: bool,
        nvme_path: str,
        offload_params_device: str,
        params_buffer_count: int,
        params_buffer_size: int,
        max_in_cpu: int,
        offload_optimizer_device: str,
        optimizer_buffer_count: int,
        pin_memory: bool,
        block_size: int,
        queue_depth: int,
        single_submit: bool,
        overlap_events: bool,
        thread_count: int,
        **zero_kwargs: Any,
    ) -> Dict:
        cfg = super()._create_default_config(
            zero_optimization,
            zero_allow_untested_optimizer,
            logging_batch_size_per_gpu,
            partition_activations,
            cpu_checkpointing,
            contiguous_memory_optimization,
            synchronize_checkpoint_boundary,
            offload_optimizer,
            offload_parameters,
            nvme_path,
            offload_params_device,
            params_buffer_count,
            params_buffer_size,
            max_in_cpu,
            offload_optimizer_device,
            optimizer_buffer_count,
            pin_memory,
            block_size,
            queue_depth,
            single_submit,
            overlap_events,
            thread_count,
            **zero_kwargs,
        )
        if zero_optimization:
            if offload_parameters:
                cfg = {
                    "zero_force_ds_cpu_optimizer": False,
                    **cfg,
                }
        return cfg

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            "deepspeed",
            cls,
            description="Default DeepSpeed Strategy",
            override=True,
        )
        strategy_registry.register(
            "deepspeed_stage_1",
            cls,
            description="DeepSpeed with ZeRO Stage 1 enabled",
            stage=1,
            override=True,
        )
        strategy_registry.register(
            "deepspeed_stage_2",
            cls,
            description="DeepSpeed with ZeRO Stage 2 enabled",
            stage=2,
            override=True,
        )
        strategy_registry.register(
            "deepspeed_stage_2_offload",
            cls,
            description="DeepSpeed ZeRO Stage 2 and CPU Offload",
            stage=2,
            offload_optimizer=True,
            override=True,
        )
        strategy_registry.register(
            "deepspeed_stage_3",
            cls,
            description="DeepSpeed ZeRO Stage 3",
            stage=3,
            override=True,
        )
        strategy_registry.register(
            "deepspeed_stage_3_offload",
            cls,
            description="DeepSpeed ZeRO Stage 3 and CPU Offload",
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            override=True,
        )
        strategy_registry.register(
            "deepspeed_stage_3_offload_nvme",
            cls,
            description="DeepSpeed ZeRO Stage 3 and NVMe Offload",
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            remote_device="nvme",
            offload_params_device="nvme",
            offload_optimizer_device="nvme",
            override=True,
        )


def apply_fsdp_strategy_patch():
    FSDPStrategy.register_strategies(pl.strategies.StrategyRegistry)


def apply_deepspeed_strategy_patch():
    DeepSpeedStrategy.register_strategies(pl.strategies.StrategyRegistry)


def apply_all_patches():
    apply_fsdp_strategy_patch()
    apply_deepspeed_strategy_patch()
