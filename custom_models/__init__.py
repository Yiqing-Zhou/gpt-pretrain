import importlib
from collections import OrderedDict

from transformers.models.auto import auto_factory, configuration_auto

CONFIG_MAPPING_NAMES = OrderedDict([])


def register_custom_configs():
    for model_type, map_name in CONFIG_MAPPING_NAMES.items():
        module_name = configuration_auto.model_type_to_module_name(model_type)
        module = importlib.import_module(f".{module_name}", "custom_models")
        mapping = getattr(module, map_name)
        configuration_auto.AutoConfig.register(model_type, mapping)


class _LazyAutoMapping(auto_factory._LazyAutoMapping):
    def _load_attr_from_module(self, model_type, attr):
        module_name = auto_factory.model_type_to_module_name(model_type)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "custom_models")
        return auto_factory.getattribute_from_module(self._modules[module_name], attr)


MODEL_MAPPING_NAMES = OrderedDict(
    [
        ("gpt2", "GPT2Model"),
    ]
)


MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        ("gpt2", "GPT2LMHeadModel"),
    ]
)


MODEL_MAPPING = _LazyAutoMapping(
    {**configuration_auto.CONFIG_MAPPING_NAMES, **CONFIG_MAPPING_NAMES}, MODEL_MAPPING_NAMES
)


MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoMapping(
    {**configuration_auto.CONFIG_MAPPING_NAMES, **CONFIG_MAPPING_NAMES}, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
)


class AutoModel(auto_factory._BaseAutoModelClass):
    _model_mapping = MODEL_MAPPING


AutoModel = auto_factory.auto_class_update(AutoModel)


class AutoModelForCausalLM(auto_factory._BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING


AutoModelForCausalLM = auto_factory.auto_class_update(AutoModelForCausalLM)
