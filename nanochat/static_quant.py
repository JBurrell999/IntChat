"""Small post-training static W8A8 prototype for NanoChat experiments.

This intentionally does not implement an integer Transformer. Linear layers use
PyTorch's existing INT8 x INT8 -> INT32 matrix multiplication; outputs are then
dequantized for NanoChat's floating-point nonlinear and residual operations.
No custom kernel is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


@dataclass
class CalibrationHandle:
    name: str
    module: nn.Linear
    handle: torch.utils.hooks.RemovableHandle
    max_abs: torch.Tensor


def _symmetric_scale(max_abs: torch.Tensor, qmax: int = 127) -> torch.Tensor:
    return (max_abs.float() / qmax).clamp_min(torch.finfo(torch.float32).tiny)


def is_int8_eligible(module: nn.Linear) -> bool:
    """Whether the existing CUDA INT8 matmul supports this layer efficiently.

    NanoChat's core projections are aligned. Tiny smear/value-gate projections
    are not and remain floating point rather than relying on private-kernel
    behavior for unaligned dimensions.
    """
    return module.in_features % 8 == 0 and module.out_features % 8 == 0


class StaticInt8Linear(nn.Module):
    """Frozen per-tensor activation/per-output-channel weight W8A8 linear."""

    def __init__(self, layer: nn.Linear, activation_max_abs: torch.Tensor):
        super().__init__()
        weight = layer.weight.detach().float()
        weight_scale = _symmetric_scale(weight.abs().amax(dim=1))
        weight_q = torch.round(weight / weight_scale[:, None]).clamp(-127, 127).to(torch.int8)
        self.register_buffer("weight_q_t", weight_q.mT.contiguous())
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer("activation_scale", _symmetric_scale(activation_max_abs))
        self.register_buffer("bias", None if layer.bias is None else layer.bias.detach().float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        x_q = torch.round(x.float() / self.activation_scale).clamp(-127, 127).to(torch.int8)
        try:
            accumulator = torch._int_mm(
                x_q.reshape(-1, input_shape[-1]).contiguous(), self.weight_q_t
            )
        except RuntimeError as error:
            raise RuntimeError(
                "This PyTorch/device build does not support the INT8 matrix multiplication "
                "required by StaticInt8Linear"
            ) from error
        output = accumulator.float() * (self.activation_scale * self.weight_scale)
        bias = None if self.bias is None else self.bias.to(x.dtype)
        output = output.reshape(*input_shape[:-1], self.weight_scale.numel()).to(x.dtype)
        return output if bias is None else output + bias


class DynamicInt8Linear(nn.Module):
    """W8A8 control with a runtime per-tensor activation scale.

    This deliberately uses floating-point range estimation. It is a control for
    whether any observed stability comes from integer matmul alone or from the
    calibration-fixed activation grid used by :class:`StaticInt8Linear`.
    """

    def __init__(self, layer: nn.Linear):
        super().__init__()
        weight = layer.weight.detach().float()
        weight_scale = _symmetric_scale(weight.abs().amax(dim=1))
        weight_q = torch.round(weight / weight_scale[:, None]).clamp(-127, 127).to(torch.int8)
        self.register_buffer("weight_q_t", weight_q.mT.contiguous())
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer("bias", None if layer.bias is None else layer.bias.detach().float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        activation_scale = _symmetric_scale(x.detach().abs().amax())
        x_q = torch.round(x.float() / activation_scale).clamp(-127, 127).to(torch.int8)
        try:
            accumulator = torch._int_mm(
                x_q.reshape(-1, input_shape[-1]).contiguous(), self.weight_q_t
            )
        except RuntimeError as error:
            raise RuntimeError(
                "This PyTorch/device build does not support the INT8 matrix multiplication "
                "required by DynamicInt8Linear"
            ) from error
        output = accumulator.float() * (activation_scale * self.weight_scale)
        bias = None if self.bias is None else self.bias.to(x.dtype)
        output = output.reshape(*input_shape[:-1], self.weight_scale.numel()).to(x.dtype)
        return output if bias is None else output + bias


def attach_linear_calibrators(model: nn.Module) -> list[CalibrationHandle]:
    """Observe each Linear input while the unmodified float model runs."""
    calibrators: list[CalibrationHandle] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or not is_int8_eligible(module):
            continue
        maximum = torch.zeros((), dtype=torch.float32, device=module.weight.device)

        def observe(mod, args, _name=name, _maximum=maximum):
            del mod, _name
            value = args[0].detach().abs().amax().float()
            _maximum.copy_(torch.maximum(_maximum, value))

        handle = module.register_forward_pre_hook(observe)
        calibrators.append(CalibrationHandle(name, module, handle, maximum))
    return calibrators


def _set_submodule(root: nn.Module, path: str, module: nn.Module) -> None:
    parent_path, _, child_name = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


def convert_calibrated_linears(model: nn.Module, calibrators: Iterable[CalibrationHandle]) -> nn.Module:
    """Remove observers and replace observed linears with frozen W8A8 layers."""
    calibrators = list(calibrators)
    if not calibrators:
        raise ValueError("no Linear modules were calibrated")
    for calibration in calibrators:
        calibration.handle.remove()
        if calibration.max_abs.item() == 0:
            raise RuntimeError(f"Linear input was never observed during calibration: {calibration.name}")
        replacement = StaticInt8Linear(calibration.module, calibration.max_abs)
        _set_submodule(model, calibration.name, replacement)
    return model


def convert_dynamic_linears(model: nn.Module) -> nn.Module:
    """Replace every float Linear with the dynamic-activation W8A8 control."""
    replacements = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and is_int8_eligible(module)
    ]
    for name, module in replacements:
        _set_submodule(model, name, DynamicInt8Linear(module))
    return model
