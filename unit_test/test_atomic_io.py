import os
import json
import tempfile
import torch
import pytest

from utils.atomic_io import atomic_torch_save, atomic_json_dump


def test_atomic_torch_save_creates_file():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.pt")
        tensor = torch.randn(3, 3)
        atomic_torch_save(tensor, path)
        assert os.path.isfile(path)
        loaded = torch.load(path, weights_only=True)
        assert torch.equal(tensor, loaded)


def test_atomic_torch_save_no_tmp_left():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.pt")
        atomic_torch_save(torch.randn(2), path)
        files = os.listdir(d)
        assert files == ["test.pt"], f"Unexpected files: {files}"


def test_atomic_torch_save_creates_parent_dirs():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sub", "dir", "test.pt")
        atomic_torch_save(torch.randn(2), path)
        assert os.path.isfile(path)


def test_atomic_json_dump_creates_file():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.json")
        data = {"key": "value", "n": 42}
        atomic_json_dump(data, path)
        assert os.path.isfile(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data


def test_atomic_json_dump_no_tmp_left():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.json")
        atomic_json_dump({"a": 1}, path)
        files = os.listdir(d)
        assert files == ["test.json"], f"Unexpected files: {files}"


def test_atomic_torch_save_overwrites_existing():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.pt")
        atomic_torch_save(torch.tensor([1.0]), path)
        atomic_torch_save(torch.tensor([2.0]), path)
        loaded = torch.load(path, weights_only=True)
        assert torch.equal(loaded, torch.tensor([2.0]))


def test_atomic_json_dump_overwrites_existing():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.json")
        atomic_json_dump({"v": 1}, path)
        atomic_json_dump({"v": 2}, path)
        with open(path) as f:
            assert json.load(f) == {"v": 2}
