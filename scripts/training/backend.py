"""Construct the CUDA resources used throughout one shard invocation."""

import torch

from scripts.coach_core.types.training import TrainingBackend
from scripts.training.engine import create_optimizer
from scripts.training.model_io import load_draft, load_target_tables
from scripts.training.objective import ShardObjective


def create_backend(args, validation):
    device = torch.device("cuda:0")
    model, reader = load_draft(args.initial_draft, device)
    embedding, head = load_target_tables(args.target_model, device)
    objective = ShardObjective(
        model, embedding, head,
        args.context, args.batch_size, validation,
        args.objective,
    )
    optimizer = create_optimizer(model, args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    return TrainingBackend(model, reader, objective, optimizer, scaler)
