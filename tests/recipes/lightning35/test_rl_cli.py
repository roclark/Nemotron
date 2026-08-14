# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the Lightning 3.5 RL CLI execution dispatch."""

from __future__ import annotations

from pathlib import Path


def test_docker_ray_uses_experiment_instead_of_ray_job(monkeypatch, tmp_path: Path) -> None:
    import nemo_run as run

    from nemotron.cli.commands.lightning35 import rl

    captured: dict = {}
    executor = object()

    def fake_create_executor(**kwargs):
        captured["executor_kwargs"] = kwargs
        return executor

    class FakeExperiment:
        def __init__(self, name: str):
            captured["experiment_name"] = name

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def add(self, task, **kwargs):
            captured["task"] = task
            captured["add_kwargs"] = kwargs

        def run(self, **kwargs):
            captured["run_kwargs"] = kwargs

    monkeypatch.setattr(rl, "create_executor", fake_create_executor)
    monkeypatch.setattr(run, "Experiment", FakeExperiment)

    rl._execute_docker_ray(
        train_path=tmp_path / "train.yaml",
        env={
            "executor": "docker",
            "container": "example.invalid/nemo-rl:test",
            "gpus_per_node": 1,
        },
        passthrough=["grpo.max_num_steps=1"],
        attached=True,
        env_vars={"WANDB_ENABLED": "false"},
        startup_commands=[],
    )

    assert captured["executor_kwargs"]["env"]["executor"] == "docker"
    assert captured["add_kwargs"]["executor"] is executor
    assert captured["add_kwargs"]["tail_logs"] is True
    assert captured["run_kwargs"] == {"detach": False}
    assert "cp main.py /opt/nemo-rl/" in captured["task"].inline
    assert "cp config.yaml /opt/nemo-rl/" in captured["task"].inline
    assert "cd /opt/nemo-rl && uv run python main.py --config config.yaml" in captured[
        "task"
    ].inline
    assert "grpo.max_num_steps=1" in captured["task"].inline
