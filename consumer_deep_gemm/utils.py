"""SM count and TC utilization controls, matching DeepGEMM's interface."""

import torch

_num_sms: int = 0
_tc_util: float = 1.0
_ignore_compile_dims: bool = False
_block_size_multiple_of: int = 1
_pdl: bool = False


def set_num_sms(num_sms: int) -> None:
    global _num_sms
    _num_sms = num_sms


def get_num_sms() -> int:
    global _num_sms
    if _num_sms == 0:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        _num_sms = props.multi_processor_count
    return _num_sms


def set_tc_util(tc_util: float) -> None:
    global _tc_util
    _tc_util = tc_util


def get_tc_util() -> float:
    return _tc_util


def set_ignore_compile_dims(v: bool) -> None:
    global _ignore_compile_dims
    _ignore_compile_dims = v


def set_block_size_multiple_of(v: int) -> None:
    global _block_size_multiple_of
    _block_size_multiple_of = v


def set_pdl(v: bool) -> None:
    global _pdl
    _pdl = v


def get_pdl() -> bool:
    return _pdl
