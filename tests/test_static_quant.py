import torch

from nanochat.static_quant import (
    DynamicInt8Linear,
    StaticInt8Linear,
    attach_linear_calibrators,
    convert_calibrated_linears,
    convert_dynamic_linears,
)


def test_static_quant_freezes_integer_weights_and_activation_scale():
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(16, 8), torch.nn.ReLU(), torch.nn.Linear(8, 8))
    calibration = attach_linear_calibrators(model)
    model(torch.randn(8, 16))
    convert_calibrated_linears(model, calibration)
    assert isinstance(model[0], StaticInt8Linear)
    assert isinstance(model[2], StaticInt8Linear)
    assert model[0].weight_q_t.dtype == torch.int8
    assert model[0].activation_scale.item() > 0


def test_unaligned_auxiliary_linear_stays_float():
    model = torch.nn.Sequential(torch.nn.Linear(24, 1, bias=False))
    calibration = attach_linear_calibrators(model)
    assert calibration == []
    convert_dynamic_linears(model)
    assert isinstance(model[0], torch.nn.Linear)


def test_static_quant_output_is_repeatable_and_close():
    torch.manual_seed(8)
    float_model = torch.nn.Linear(16, 8, bias=False)
    test_input = torch.randn(4, 16)
    expected = float_model(test_input)
    model = torch.nn.Sequential(float_model)
    calibration = attach_linear_calibrators(model)
    model(torch.randn(32, 16) * 2)
    convert_calibrated_linears(model, calibration)
    first = model(test_input)
    second = model(test_input)
    assert torch.equal(first, second)
    torch.testing.assert_close(first, expected, atol=0.04, rtol=0.08)


def test_tiny_nanochat_model_can_be_calibrated_and_converted(monkeypatch):
    # On Hopper machines nanochat auto-selects CUDA-only FlashAttention 3 at
    # import time. This test intentionally keeps its tiny model on CPU, so force
    # the portable SDPA path for this test only.
    import nanochat.flash_attention as flash_attention_module
    from nanochat.gpt import GPT, GPTConfig

    monkeypatch.setattr(flash_attention_module, "USE_FA3", False)
    config = GPTConfig(sequence_len=8, vocab_size=32, n_layer=1, n_head=2, n_kv_head=1, n_embd=24)
    model = GPT(config)
    model.init_weights()
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    calibration = attach_linear_calibrators(model)
    model(ids)
    convert_calibrated_linears(model, calibration)
    first = model(ids)
    second = model(ids)
    assert first.shape == (1, 3, 32)
    assert torch.equal(first, second)
    assert any(isinstance(module, StaticInt8Linear) for module in model.modules())


def test_dynamic_int8_control_uses_integer_weights():
    torch.manual_seed(9)
    float_layer = torch.nn.Linear(16, 8, bias=False)
    model = torch.nn.Sequential(float_layer)
    x = torch.randn(4, 16)
    expected = model(x)
    convert_dynamic_linears(model)
    assert isinstance(model[0], DynamicInt8Linear)
    assert model[0].weight_q_t.dtype == torch.int8
    torch.testing.assert_close(model(x), expected, atol=0.04, rtol=0.08)
