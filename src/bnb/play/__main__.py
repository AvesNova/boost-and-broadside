"""`python -m bnb.play`: watch, play or replay a match, in a window or offscreen."""

import torch
import tyro

from .app import Options, run_offscreen


def main() -> None:
    options = tyro.cli(Options)
    torch.set_num_threads(options.threads)
    if options.offscreen is not None:
        paths = run_offscreen(options)
        print(f"wrote {len(paths)} frames to {options.offscreen}")
        return
    from .window import run_window

    run_window(options)


if __name__ == "__main__":
    main()
