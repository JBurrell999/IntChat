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
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))
    calibration = attach_linear_calibrators(model)
    model(torch.randn(8, 4))
    convert_calibrated_linears(model, calibration)
    assert isinstance(model[0], StaticInt8Linear)
    assert isinstance(model[2], StaticInt8Linear)
    assert model[0].weight_q_t.dtype == torch.int8
    assert model[0].activation_scale.item() > 0


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


def test_tiny_nanochat_model_can_be_calibrated_and_converted():
    from nanochat.gpt import GPT, GPTConfig

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
