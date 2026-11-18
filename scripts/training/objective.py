"""DFlash loss computation and weighted validation on the fixed monitor."""

import torch
from torch.nn import functional as F

from scripts.coach_core.constants.recipe import (
    DPACE_OBJECTIVE,
    FEATURE_WIDTH,
    MASK_TOKEN_ID,
    SELECTOR_IGNORE_INDEX,
    SELECTOR_LOSS_WEIGHT,
    SELECTOR_TOP_K,
    UNIFORM_OBJECTIVE,
    VOCAB_SIZE,
)
from scripts.coach_core.metrics import weighted_metrics
from scripts.coach_core.types.training import ObjectiveBatch
from scripts.training.data import copy_feature_rows
from scripts.training.position_loss import dpace_loss


class ShardObjective:
    def __init__(
        self, model, embedding, head, context, batch_size, validation,
        objective=UNIFORM_OBJECTIVE,
    ):
        self.model = model
        self.embedding = embedding
        self.head = head
        self.context = context
        self.batch_size = batch_size
        self.validation = validation
        self.objective = objective
        self.device = embedding.device

    def prepare_batch(self, batch, anchors, width):
        b = len(batch)
        ctx = self.context
        x = torch.zeros((b, ctx, FEATURE_WIDTH), device=self.device, dtype=torch.float16)
        noise = torch.full((b, width), MASK_TOKEN_ID, device=self.device, dtype=torch.long)
        positions = torch.zeros((b, ctx + width), device=self.device, dtype=torch.long)
        attention = torch.ones(
            (b, 1, width, ctx + width),
            device=self.device,
            dtype=torch.bool,
        )
        gold = []
        previous = []
        for index, (sample, anchor) in enumerate(zip(batch, anchors, strict=False)):
            begin = max(0, anchor - ctx)
            length = anchor - begin
            pad = ctx - length
            x[index, pad:] = torch.from_numpy(copy_feature_rows(sample, begin, anchor)).to(
                device=self.device, dtype=torch.float16
            )
            positions[index, pad:] = torch.arange(begin, anchor + width, device=self.device)
            attention[index, :, :, :pad] = False
            ids = sample["input_ids"]
            noise[index, 0] = ids[anchor]
            gold.append(ids[anchor + 1 : anchor + width])
            previous.append(ids[anchor : anchor + width - 1])
        gold = torch.tensor(gold, device=self.device)
        previous = torch.tensor(previous, device=self.device)
        return ObjectiveBatch(
            features=x,
            noise=noise,
            positions=positions,
            attention=attention,
            gold=gold,
            previous=previous,
        )

    def compute(self, batch, anchors, width, *, training=False):
        prepared = self.prepare_batch(batch, anchors, width)
        with torch.autocast("cuda", dtype=torch.float16):
            hidden = self.model(
                position_ids=prepared.positions,
                attention_mask=prepared.attention,
                noise_embedding=F.embedding(prepared.noise, self.embedding),
                target_hidden=prepared.features,
            )
            logits = F.linear(hidden[:, 1:], self.head).float()
            token_nll = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), prepared.gold.flatten(), reduction="none",
            ).reshape_as(prepared.gold)
            unary, candidates = logits.topk(SELECTOR_TOP_K, dim=-1)
            selector = self.model.candidate_selector
            gate = selector.hidden_projection(hidden[:, 1:]).float()
            predecessor = selector.predecessor_codebook(prepared.previous).float()
            successor = selector.successor_codebook(candidates).float()
            weighted_predecessor = (predecessor * gate)[:, :, None, :]
            scores = unary + (weighted_predecessor * successor).sum(-1)
            matches = candidates.eq(prepared.gold[:, :, None])
            covered = matches.any(-1)
            target_indices = matches.long().argmax(-1).masked_fill(~covered, SELECTOR_IGNORE_INDEX)
            selector_nll = F.cross_entropy(
                scores.transpose(1, 2), target_indices, reduction="none",
                ignore_index=SELECTOR_IGNORE_INDEX,
            )
            ce = token_nll.mean()
            selector_loss = selector_nll.sum() / covered.sum().clamp_min(1)
            if training and self.objective == DPACE_OBJECTIVE:
                loss = dpace_loss(token_nll, selector_nll)
            else:
                loss = ce + SELECTOR_LOSS_WEIGHT * selector_loss
        chosen = candidates.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
        correct = chosen.eq(prepared.gold)
        prefix = correct.long().cumprod(-1).sum(-1).float()
        return loss, {
            "ce": float(ce.detach()),
            "selector_ce": float(selector_loss.detach()),
            "accuracy": float(correct.float().mean()),
            "prefix_tokens": float(prefix.mean()),
            "top16_recall": float(covered.float().mean()),
        }

    def evaluate(self, full=True):
        self.model.eval()
        chosen = self.validation if full else self.validation[: min(
            16,
            len(self.validation),
        )]
        records = []
        with torch.no_grad():
            for start in range(0, len(chosen), self.batch_size):
                batch = chosen[start : start + self.batch_size]
                for fraction in (0.25, 0.7):
                    anchors = self.validation_anchors(batch, fraction)
                    loss, metrics = self.compute(batch, anchors, 8)
                    records.append({
                        "loss": float(loss.detach()),
                        **metrics,
                        "samples": len(batch),
                    })
        return weighted_metrics(records)

    def validation_anchors(self, batch, fraction):
        anchors = []
        for sample in batch:
            prompt_length = sample["prompt_n"]
            generated = len(sample["input_ids"]) - prompt_length - 8
            anchors.append(prompt_length + int(generated * fraction))
        return anchors
