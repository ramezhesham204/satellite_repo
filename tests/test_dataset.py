import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import build_model


def test_cnn_output_shape():
    import torch
    m = build_model("cnn", num_classes=10)
    x = torch.randn(2, 3, 64, 64)
    out = m(x)
    assert out.shape == (2, 10)
