"""PERD's pre-publish build checks, run locally.

The describe step reads perd.toml, imports workflow.py and compiles its
contract, then dry-runs page.py against it. A failure here is a failed
publish there.
"""

import asyncio
import tomllib
from pathlib import Path

from perd_worker.compiler import compile_contract

import workflow

ROOT = Path(__file__).resolve().parents[1]

#: The build's perd.toml schema (perd_api.workflow_manifest).
ALLOWED_CLASSES = {"small", "medium", "large", "gpu_t4"}
WORKFLOW_KEYS = {"summary": str, "tags": list, "category": str, "license": str, "homepage": str}

#: The interface existing workspaces and pages call: (name, type, default).
TRAIN_PARAMS = [
    ("batch_size", int, 4),
    ("epochs", int, 100),
    ("output_dim", int, 6),
    ("num_pde_constraints", int, 5),
    ("d_model", int, 64),
    ("num_heads", int, 4),
    ("dim_feedforward", int, 128),
    ("num_layers", int, 2),
    ("dropout", float, 0.1),
    ("learning_rate", float, 0.001),
    ("lambda_data", float, 1.0),
    ("lambda_pde", float, 1.0),
    ("report_every", int, 10),
]


def test_perd_toml_passes_the_build_schema():
    data = tomllib.loads((ROOT / "perd.toml").read_text())
    assert set(data) <= {"compute", "workflow"}
    assert set(data["compute"]) == {"class"}
    assert data["compute"]["class"] in ALLOWED_CLASSES
    meta = data["workflow"]
    assert set(meta) <= set(WORKFLOW_KEYS)
    for key, value in meta.items():
        assert isinstance(value, WORKFLOW_KEYS[key]), key
    assert len(meta["summary"]) <= 500
    assert len(meta["tags"]) <= 12 and all(len(tag) <= 32 for tag in meta["tags"])
    assert all(len(meta[key]) <= 200 for key in ("category", "license", "homepage"))


def test_requirements_leave_perd_worker_to_the_build():
    lines = (ROOT / "requirements.txt").read_text().splitlines()
    packages = [line for line in lines if line.strip() and not line.lstrip().startswith(("#", "-"))]
    assert not any(line.lower().startswith(("perd-worker", "perd_worker")) for line in packages)


def test_the_operations_compile_and_describe_as_on_perd():
    from perd.v1 import worker_pb2
    from perd_worker.runtime import WorkerServicer

    contract = compile_contract("dca_pinn", workflow.workflow)
    servicer = WorkerServicer(contract, workflow.workflow)
    described = asyncio.run(servicer.Describe(worker_pb2.DescribeRequest(), None))
    assert described.contract_hash == contract.contract_hash
    assert [op.function_name for op in described.contract.operations] == ["train"]


def test_train_keeps_its_published_interface():
    op = workflow.workflow.ops["train"]
    assert op.mode == "bi_di"
    assert op.input_item_types == (float,) * 7
    assert op.output_item_types == (int, float)
    assert [(name, kind, default) for name, kind, _, default in op.params] == TRAIN_PARAMS


def test_the_page_renders():
    from perdlit._script import check_page

    summary = check_page("page", "workflow")
    assert [button["label"] for button in summary["buttons"]] == ["Train"]
    assert summary["registered_ops"] == ["train"]
