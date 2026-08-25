import torch
from src.model import get_model


def test_resnet18_output_shape():
    model = get_model("resnet18", 10)
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    assert output.shape == (2, 10)
