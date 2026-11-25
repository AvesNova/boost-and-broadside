"""`bnb play` (or `python -m bnb.play`): watch, play or replay a match, in a window or offscreen."""

import torch
import tyro

from .app import Options, run_offscreen


def main(arguments: list[str] | None = None) -> None:
    options = tyro.cli(Options, args=arguments, prog="bnb play")
    if options.replay is None:
        try:
            options.game()  # an unknown preset or map stops here, naming the known ones
        except ValueError as error:
            raise SystemExit(str(error)) from error
    torch.set_num_threads(options.threads)
    if options.offscreen is not None:
        paths = run_offscreen(options)
        print(f"wrote {len(paths)} frames to {options.offscreen}")
        return
    from .window import run_window

    run_window(options)


if __name__ == "__main__":
    main()
