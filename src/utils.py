"""
Common utilities for Boost and Broadside
"""

import os
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Callable, Any
import signal


def setup_directories(run_name: str, base_dir: str = "") -> Tuple[Path, Path]:
    """
    Setup output directories with consistent naming

    Args:
        run_name: Name for the run
        base_dir: Base directory for outputs

    Returns:
        Tuple of (checkpoint_dir, log_dir)
    """
    # Use centralized directories
    checkpoints_base = Path.cwd() / "checkpoints"
    logs_base = Path.cwd() / "logs"

    if base_dir:
        checkpoint_dir = Path(base_dir) / f"{run_name}_checkpoints"
        log_dir = Path(base_dir) / f"{run_name}_logs"
    else:
        checkpoint_dir = checkpoints_base / f"{run_name}_checkpoints"
        log_dir = logs_base / f"{run_name}_logs"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    return checkpoint_dir, log_dir


def setup_logging(
    run_name: str,
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Setup consistent logging across all scripts

    Args:
        run_name: Name for the run
        level: Logging level
        log_dir: Directory to save log files
        console: Whether to log to console

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(run_name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{run_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        # If no log_dir specified, create in centralized logs directory
        logs_base = Path.cwd() / "logs"
        logs_base.mkdir(parents=True, exist_ok=True)
        log_file = logs_base / f"{run_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Add a method to properly close handlers
    def close_handlers():
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    logger.close_handlers = close_handlers

    return logger


class InterruptHandler:
    """
    Context manager for handling keyboard interrupts consistently
    """

    def __init__(self, message: str = "Operation interrupted by user"):
        self.message = message
        self.original_handlers = []

    def __enter__(self):
        # Store original handlers
        self.original_handlers = signal.getsignal(signal.SIGINT)

        # Set new handler
        def handler(signum, frame):
            print(f"\n{self.message}")
            sys.exit(1)

        signal.signal(signal.SIGINT, handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original handlers
        signal.signal(signal.SIGINT, self.original_handlers)


def handle_interrupt(handler_func: Optional[Callable] = None):
    """
    Decorator for consistent interrupt handling

    Args:
        handler_func: Function to call on interrupt
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                if handler_func:
                    handler_func()
                else:
                    print("\nOperation interrupted by user")
                    sys.exit(1)

        return wrapper

    return decorator


def generate_run_name(prefix: str, suffix: Optional[str] = None) -> str:
    """
    Generate a consistent run name with timestamp

    Args:
        prefix: Prefix for the run name
        suffix: Optional suffix for the run name

    Returns:
        Generated run name
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if suffix:
        return f"{prefix}_{suffix}_{timestamp}"
    else:
        return f"{prefix}_{timestamp}"


def save_config_copy(config: dict, checkpoint_dir: Path) -> None:
    """
    Save a copy of the configuration to the checkpoint directory

    Args:
        config: Configuration dictionary
        checkpoint_dir: Directory to save the config to
    """
    import yaml

    config_copy_path = checkpoint_dir / "config.yaml"
    with open(config_copy_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"Configuration saved to: {config_copy_path}")


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary

    Args:
        path: Path to ensure exists

    Returns:
        The path (created if it didn't exist)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_latest_file(directory: Path, pattern: str = "*") -> Optional[Path]:
    """
    Get the latest file in a directory matching a pattern

    Args:
        directory: Directory to search
        pattern: Pattern to match

    Returns:
        Path to the latest file, or None if no files found
    """
    files = list(directory.glob(pattern))
    if not files:
        return None

    return max(files, key=lambda f: f.stat().st_mtime)


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds as a human-readable string

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.2f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {remaining_minutes}m {remaining_seconds:.2f}s"


def format_bytes(bytes_value: int) -> str:
    """
    Format a byte count as a human-readable string

    Args:
        bytes_value: Number of bytes

    Returns:
        Formatted byte string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


class ProgressTracker:
    """
    Simple progress tracking utility
    """

    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()

    def update(self, increment: int = 1) -> None:
        """Update progress"""
        self.current += increment
        self._print_progress()

    def set_current(self, current: int) -> None:
        """Set current progress"""
        self.current = current
        self._print_progress()

    def _print_progress(self) -> None:
        """Print progress to console"""
        percent = (self.current / self.total) * 100
        elapsed = (datetime.now() - self.start_time).total_seconds()

        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = format_duration(eta)
        else:
            eta_str = "unknown"

        print(
            f"\r{self.description}: {self.current}/{self.total} ({percent:.1f}%) - ETA: {eta_str}",
            end="",
        )

        if self.current >= self.total:
            print()  # New line when complete


def create_progress_bar(
    iterable, desc: Optional[str] = None, disable: bool = False, **kwargs
):
    """
    Create a progress bar using tqdm

    Args:
        iterable: Iterable to wrap
        desc: Description for the progress bar
        disable: Whether to disable the progress bar
        **kwargs: Additional arguments for tqdm

    Returns:
        Progress bar
    """
    try:
        from tqdm import tqdm

        if desc is not None:
            kwargs["desc"] = desc
        if disable:
            kwargs["disable"] = True
        return tqdm(iterable, **kwargs)
    except ImportError:
        # Fallback to simple progress tracker if tqdm not available
        if disable:
            # Return a no-op progress tracker
            class NoOpProgressTracker:
                def __init__(self, iterable):
                    self.iterable = iterable

                def __iter__(self):
                    return iter(self.iterable)

                def update(self, increment: int = 1) -> None:
                    pass

                def set_current(self, current: int) -> None:
                    pass

            return NoOpProgressTracker(iterable)
        else:
            return ProgressTracker(
                len(iterable) if hasattr(iterable, "__len__") else 0, desc or "Progress"
            )


def validate_file_exists(file_path: str, description: str = "File") -> Path:
    """
    Validate that a file exists and return the Path object

    Args:
        file_path: Path to the file
        description: Description for error messages

    Returns:
        Path object for the file

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {file_path}")
    return path


def validate_directory_exists(dir_path: str, description: str = "Directory") -> Path:
    """
    Validate that a directory exists and return the Path object

    Args:
        dir_path: Path to the directory
        description: Description for error messages

    Returns:
        Path object for the directory

    Raises:
        NotADirectoryError: If the path is not a directory
    """
    path = Path(dir_path)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {dir_path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{description} is not a directory: {dir_path}")
    return path
