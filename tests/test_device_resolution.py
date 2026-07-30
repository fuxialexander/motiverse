from __future__ import annotations

import pytest

from motiverse import dense_subset_workflow


def test_resolve_analysis_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(
        dense_subset_workflow.torch.cuda,
        "is_available",
        lambda: False,
    )

    device, metadata = dense_subset_workflow.resolve_analysis_device("auto")

    assert device == "cpu"
    assert metadata["requested_device"] == "auto"
    assert metadata["effective_device"] == "cpu"
    assert metadata["cuda_available"] is False
    assert metadata["device_resolution"] == "auto_cuda_unavailable_cpu"


def test_resolve_analysis_device_rejects_unavailable_cuda(monkeypatch):
    monkeypatch.setattr(
        dense_subset_workflow.torch.cuda,
        "is_available",
        lambda: False,
    )

    with pytest.raises(ValueError, match="torch.cuda.is_available"):
        dense_subset_workflow.resolve_analysis_device("cuda")
