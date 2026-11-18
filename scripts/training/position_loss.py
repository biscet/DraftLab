"""Position-weighted D-PACE objective for a batch of DFlash blocks."""

from torch import Tensor

from scripts.coach_core.constants.recipe import DPACE_ALPHA, SELECTOR_LOSS_WEIGHT


def dpace_loss(token_nll: Tensor, selector_nll: Tensor) -> Tensor:
    """Weight both losses by detached prefix confidence and average over blocks.

    Selector NLL must be zero at positions where the gold token is absent
    from the candidate set.
    """
    confidence = DPACE_ALPHA + (1 - DPACE_ALPHA) * (-token_nll.detach()).exp()
    prefix = confidence.cumprod(dim=-1)
    weights = prefix.flip(-1).cumsum(dim=-1).flip(-1)
    weighted = weights * (token_nll + SELECTOR_LOSS_WEIGHT * selector_nll)
    return weighted.sum(dim=-1).mean()
