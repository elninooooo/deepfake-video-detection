import torch

from modelsgenerate.tim_extractor import TIMExtractor


def test_shape():
    x = torch.rand(2, 16, 3, 224, 224)
    out = TIMExtractor()(x)
    assert out.shape == (2, 15, 3, 224, 224)


def test_nonneg():
    x = torch.rand(1, 4, 3, 32, 32)
    out = TIMExtractor()(x)
    assert (out >= 0).all()
