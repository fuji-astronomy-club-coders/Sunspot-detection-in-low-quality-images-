"""Provide application version and optional Git revision metadata.

The application version is maintained in the repository-root VERSION file.
Git metadata is supplemental and must never be required for image processing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSION_PATH = PROJECT_ROOT / "VERSION"
APPLICATION_NAME = "Sunspot Detection in Low-Quality Images"


def get_application_version() -> str:
    """Read the application version from the repository-root VERSION file.

    Returns
    -------
    str
        The configured application version, or "unknown" if unavailable.
    """
    try:
        version_value = VERSION_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"

    return version_value or "unknown"


def _run_git(
    project_root: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a Git command against the project root.

    Parameters
    ----------
    project_root : pathlib.Path
        Directory containing the application's Git working tree.
    *arguments : str
        Git command and arguments.
    check : bool, default=True
        Whether a non-zero exit code should raise CalledProcessError.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Captured Git command result.
    """
    return subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=5,
    )


def collect_git_metadata(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Collect the checked-out commit and working-tree metadata.

    The function deliberately does not fetch from a remote. It records the HEAD
    available in the local checkout at application startup.

    Parameters
    ----------
    project_root : pathlib.Path, default=PROJECT_ROOT
        Repository root to inspect.

    Returns
    -------
    dict[str, Any]
        Git version, branch, HEAD commit, and working-tree state. When Git
        metadata cannot be collected, "available" is false and "reason"
        explains why; callers can continue without Git information.
    """
    unavailable: dict[str, Any] = {
        "available": False,
        "git_version": None,
        "branch": None,
        "commit": None,
        "working_tree": "unknown",
        "reason": None,
    }

    try:
        git_version = _run_git(project_root, "--version").stdout.strip()
        repository = _run_git(
            project_root,
            "rev-parse",
            "--show-toplevel",
            check=False,
        )
        if repository.returncode != 0:
            unavailable["reason"] = "not a Git working tree"
            return unavailable

        commit_output = _run_git(
            project_root,
            "log",
            "-1",
            "--format=%H%x1f%cI%x1f%s",
        ).stdout.strip()
        commit_parts = commit_output.split("\x1f", maxsplit=2)
        if len(commit_parts) != 3 or not commit_parts[0]:
            unavailable["reason"] = "HEAD commit metadata is unavailable"
            return unavailable

        branch_result = _run_git(
            project_root,
            "symbolic-ref",
            "--quiet",
            "--short",
            "HEAD",
            check=False,
        )
        status = _run_git(
            project_root,
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ).stdout

        return {
            "available": True,
            "git_version": git_version,
            "branch": branch_result.stdout.strip() or "detached",
            "commit": {
                "sha": commit_parts[0],
                "short_sha": commit_parts[0][:12],
                "date": commit_parts[1],
                "subject": commit_parts[2],
            },
            "working_tree": "dirty" if status.strip() else "clean",
            "reason": None,
        }
    except FileNotFoundError:
        unavailable["reason"] = "Git executable not found"
    except subprocess.TimeoutExpired:
        unavailable["reason"] = "Git command timed out"
    except subprocess.CalledProcessError:
        unavailable["reason"] = "Git command failed"
    except OSError as error:
        unavailable["reason"] = f"Git metadata unavailable ({type(error).__name__})"

    return unavailable
