"""
Filesystem Path Security and Sanitization Module for Cheating Surveillance System.

Provides robust validation and containment helpers to prevent path traversal,
arbitrary file overwrites, Windows device-name collisions, and unintended
symlink or junction redirection for application-generated output files.
"""

import os
from pathlib import Path
import stat
from typing import Collection, Optional, Union

WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}

ILLEGAL_FILENAME_CHARS = set('<>:"/\\|?*')


def is_symlink_or_junction(path: Union[str, Path]) -> bool:
    """
    Checks whether the target path is a symlink or Windows directory junction / reparse point.
    Works reliably across POSIX and Windows (including Python 3.12+ where os.path.islink
    does not detect NTFS directory junctions).
    """
    p = Path(path)
    if p.is_symlink():
        return True
    if hasattr(os.path, "isjunction") and os.path.isjunction(p):
        return True
    try:
        lst = os.lstat(p)
        if hasattr(lst, "st_file_attributes") and (lst.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            return True
    except (OSError, ValueError):
        pass
    return False


def validate_safe_filename(
    filename: str,
    allow_extensions: Optional[Collection[str]] = None,
) -> str:
    """
    Validates that a filename string is a safe, single-component leaf filename.
    Rejects path traversal sequences, absolute paths, path separators, illegal
    characters, and Windows reserved device names.

    Args:
        filename: Filename string to validate.
        allow_extensions: Optional collection of permitted extensions (e.g. {'.csv', '.json'}).

    Returns:
        str: Validated filename.

    Raises:
        ValueError: If the filename fails any security check.
    """
    if not isinstance(filename, str):
        raise ValueError(f"Filename must be a string, got {type(filename).__name__}")

    s = filename.strip()
    if not s:
        raise ValueError("Filename cannot be empty or whitespace-only.")

    if "/" in s or "\\" in s:
        raise ValueError(f"Path traversal detected: filename '{filename}' cannot contain path separators.")

    if s in (".", "..") or ".." in s:
        raise ValueError(f"Path traversal detected: filename '{filename}' cannot contain '..'.")

    if any(c in ILLEGAL_FILENAME_CHARS or ord(c) < 32 for c in s):
        raise ValueError(f"Filename '{filename}' contains invalid filesystem characters.")

    stem = Path(s).stem.upper()
    if stem in WINDOWS_RESERVED_NAMES or s.upper() in WINDOWS_RESERVED_NAMES:
        raise ValueError(f"Filename '{filename}' uses a reserved system device name.")

    if allow_extensions is not None:
        ext = Path(s).suffix.lower()
        allowed = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in allow_extensions}
        if ext not in allowed:
            raise ValueError(f"File extension '{ext}' not in allowed extensions: {allowed}")

    return s


def resolve_safe_output_path(
    base_dir: Union[str, Path],
    filename: str,
    allow_extensions: Optional[Collection[str]] = None,
    allow_symlinks: bool = False,
) -> Path:
    """
    Resolves a base directory and a leaf filename, guaranteeing that the target
    cannot escape the base directory via traversal or link redirection.

    Args:
        base_dir: Intended parent directory.
        filename: Leaf filename to append.
        allow_extensions: Optional collection of permitted extensions.
        allow_symlinks: If False, rejects symlinks and junctions on base_dir and target.

    Returns:
        Path: Resolved, verified safe canonical path.

    Raises:
        ValueError: If validation fails or path containment is breached.
    """
    safe_name = validate_safe_filename(filename, allow_extensions=allow_extensions)
    base = Path(base_dir).resolve()

    if not allow_symlinks and is_symlink_or_junction(base_dir):
        raise ValueError(f"Base output directory '{base_dir}' cannot be a symlink or junction.")

    target = (base / safe_name).resolve()
    if not target.is_relative_to(base) or target.parent != base:
        raise ValueError(f"Path traversal attempt: '{filename}' escapes target directory '{base}'.")

    if target.exists() and not allow_symlinks and is_symlink_or_junction(target):
        raise ValueError(f"Target output file '{target}' cannot be a symlink or junction.")

    return target


def ensure_safe_directory(
    dir_path: Union[str, Path],
    allow_symlinks: bool = False,
) -> Path:
    """
    Ensures that a directory exists, creating parents if necessary, while verifying
    that it is not a symlink or Windows junction.

    Args:
        dir_path: Target directory path.
        allow_symlinks: If False, rejects symlinks and junctions.

    Returns:
        Path: Resolved directory Path.

    Raises:
        ValueError: If symlinks/junctions are detected when not permitted.
    """
    p = Path(dir_path)
    if p.exists() and not allow_symlinks and is_symlink_or_junction(p):
        raise ValueError(f"Directory '{dir_path}' cannot be a symlink or junction.")

    p.mkdir(parents=True, exist_ok=True)
    resolved = p.resolve()

    if not allow_symlinks and is_symlink_or_junction(resolved):
        raise ValueError(f"Resolved directory '{resolved}' cannot be a symlink or junction.")

    return resolved


def validate_safe_read_path(path: Union[str, Path]) -> Path:
    """
    Validates a read-only input file path (such as a user-selected video file).
    Confirms that the target exists and is a regular file. Does not restrict location
    to the project directory so users can provide video inputs from arbitrary drives.

    Args:
        path: Path to validate.

    Returns:
        Path: Validated Path object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a regular file.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found at '{path}'.")
    if not p.is_file():
        raise ValueError(f"Input path '{path}' is not a regular file.")
    return p
