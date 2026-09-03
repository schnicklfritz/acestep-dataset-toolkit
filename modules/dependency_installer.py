"""
Optional dependency installer for the ACE-Step Dataset Toolkit.

This module checks whether an optional Python package is available in the
same interpreter running the app and, with user approval from the UI, installs
it through:

    sys.executable -m pip install <package>

Do not call this directly on the UI thread for lengthy installs. The caller
should use a QThread worker or otherwise provide an appropriate UI warning.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class DependencyInstallResult:
    """Result returned by ensure_package()."""

    available: bool
    installed_now: bool
    message: str
    stdout: str = ""
    stderr: str = ""


def package_is_available(import_name: str) -> bool:
    """
    Return True if Python can locate an importable module.

    Example:
        package_is_available("kaggle")
    """
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def install_package(
    package_name: str,
    timeout: int = 180,
    upgrade: bool = True,
) -> DependencyInstallResult:
    """
    Install a package into the Python interpreter currently running the app.

    `package_name` is a fixed developer-controlled PyPI package spec, such as:
        "kaggle"
        "kaggle>=1.6"

    Do not pass user-entered text to this function.
    """
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
    ]

    if upgrade:
        command.append("--upgrade")

    command.append(package_name)

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return DependencyInstallResult(
            available=False,
            installed_now=False,
            message=(
                f"Timed out while installing '{package_name}' after "
                f"{timeout} seconds."
            ),
        )
    except OSError as error:
        return DependencyInstallResult(
            available=False,
            installed_now=False,
            message=(
                f"Could not start pip using this Python interpreter:\n"
                f"{sys.executable}\n\n{error}"
            ),
        )

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()

    if result.returncode != 0:
        details = stderr or stdout or "pip returned no diagnostic output."

        return DependencyInstallResult(
            available=False,
            installed_now=False,
            message=(
                f"Could not install '{package_name}'.\n\n"
                f"{details[-4000:]}"
            ),
            stdout=stdout,
            stderr=stderr,
        )

    return DependencyInstallResult(
        available=True,
        installed_now=True,
        message=f"Installed '{package_name}' successfully.",
        stdout=stdout,
        stderr=stderr,
    )


def ensure_package(
    import_name: str,
    package_name: str | None = None,
    timeout: int = 180,
) -> DependencyInstallResult:
    """
    Return success immediately if an import is already available.

    Otherwise install the declared package into the app's active Python
    environment and verify that the requested import becomes available.

    Example:
        result = ensure_package("kaggle", "kaggle")
    """
    package_name = package_name or import_name

    if package_is_available(import_name):
        return DependencyInstallResult(
            available=True,
            installed_now=False,
            message=f"'{package_name}' is already installed.",
        )

    install_result = install_package(
        package_name=package_name,
        timeout=timeout,
        upgrade=True,
    )

    if not install_result.available:
        return install_result

    if not package_is_available(import_name):
        return DependencyInstallResult(
            available=False,
            installed_now=True,
            message=(
                f"'{package_name}' was installed, but Python cannot import "
                f"'{import_name}' yet. Restart the application and try again."
            ),
            stdout=install_result.stdout,
            stderr=install_result.stderr,
        )

    return DependencyInstallResult(
        available=True,
        installed_now=True,
        message=f"'{package_name}' is installed and ready to use.",
        stdout=install_result.stdout,
        stderr=install_result.stderr,
    )
