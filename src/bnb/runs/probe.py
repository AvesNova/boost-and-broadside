"""The memory probe (runs spec 7): the run's first update, measured. Its allocator peaks go
into run.json, and a reserved peak over the budget (the device's memory less a margin)
stops the run before it trains further. One measurement of the configured run: no search
over sizes, no cache per machine. On the CPU it records nothing and never stops a run."""

import torch

MARGIN = 0.5 * 2**30  # bytes the budget leaves free on the device


class MemoryBudgetError(RuntimeError):
    """The run's first update reserved more device memory than the budget."""


def begin(device: torch.device) -> None:
    """Start the measurement: clear the allocator's peaks."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def measure(device: torch.device, margin: float = MARGIN) -> dict | None:
    """The peaks since `begin`, in bytes: allocated and reserved, with the device's total
    memory and the budget; None on the CPU."""
    if device.type != "cuda":
        return None
    total = torch.cuda.get_device_properties(device).total_memory
    return dict(
        allocated=torch.cuda.max_memory_allocated(device),
        reserved=torch.cuda.max_memory_reserved(device),
        total=total,
        budget=int(total - margin),
    )


def check(peaks: dict | None, sizes: str) -> None:
    """Raise MemoryBudgetError when the reserved peak exceeds the budget; `sizes` names
    what set them (environments, micro-batch rows)."""
    if peaks is not None and peaks["reserved"] > peaks["budget"]:
        raise MemoryBudgetError(
            f"the first update reserved {peaks['reserved'] / 2**30:.2f} GB, over the budget of "
            f"{peaks['budget'] / 2**30:.2f} GB ({peaks['total'] / 2**30:.2f} GB less the "
            f"margin), at {sizes}: lower them"
        )


def describe(peaks: dict | None) -> str:
    """One line of the peaks for a person."""
    if peaks is None:
        return "memory probe: CPU run, nothing measured"
    gigabytes = {name: value / 2**30 for name, value in peaks.items()}
    return (
        f"memory probe: allocated {gigabytes['allocated']:.2f} GB, reserved "
        f"{gigabytes['reserved']:.2f} GB, budget {gigabytes['budget']:.2f} GB of "
        f"{gigabytes['total']:.2f} GB"
    )
