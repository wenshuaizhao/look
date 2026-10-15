"""LOOK research utilities; heavy model dependencies load only when requested."""
from importlib import import_module

_MODULES = {
    'Geometry': 'geometry', 'LlmPosOffset': 'positions', 'PosOverride': 'positions',
    'assert_rows_in_full': 'positions', 'PROMPTS': 'prompts',
    'sample_prompt': 'prompts', 'LookSample': 'sample',
    'make_look_sample': 'sample', 'sample_window': 'sample',
}
__all__ = list(_MODULES) + ['ALL_PROMPTS']


def __getattr__(name):
    if name == 'ALL_PROMPTS':
        return import_module('.prompts', __name__).ALL
    if name in _MODULES:
        return getattr(import_module('.' + _MODULES[name], __name__), name)
    raise AttributeError(name)
