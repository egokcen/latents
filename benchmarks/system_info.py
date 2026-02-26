"""System information collection for benchmark reproducibility."""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone

import numpy as np
import psutil
import scipy

import latents

_BYTES_PER_GB = 1024**3


def get_system_info() -> dict:
    """Collect system hardware and software information.

    Returns
    -------
    dict
        System information with keys: timestamp, hardware, software, os.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hardware": _get_hardware_info(),
        "software": _get_software_info(),
        "os": _get_os_info(),
    }


def _get_hardware_info() -> dict:
    """Collect hardware information."""
    return {
        "cpu": _get_cpu_model(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / _BYTES_PER_GB, 1),
    }


def _get_cpu_model() -> str:
    """Get CPU model string.

    On Linux, reads from /proc/cpuinfo for detailed model name.
    Falls back to platform.processor() on other systems.
    """
    # On Linux, platform.processor() often returns just the architecture.
    # Try /proc/cpuinfo for more detail.
    if platform.system() == "Linux" and os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()

    return platform.processor() or "Unknown"


def _get_software_info() -> dict:
    """Collect software version information."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "latents": latents.__version__,
    }


def _get_os_info() -> dict:
    """Collect operating system information."""
    info = {
        "system": platform.system(),
        "release": platform.release(),
    }

    # Add distro info on Linux (Python 3.10+)
    try:
        os_release = platform.freedesktop_os_release()
        info["distro"] = os_release.get("PRETTY_NAME")
    except OSError:
        pass  # Not available (non-Linux or missing /etc/os-release)

    return info


def save_system_info(path: str | os.PathLike[str]) -> None:
    """Save system information to a JSON file.

    Parameters
    ----------
    path : str or PathLike
        Output file path.
    """
    info = get_system_info()
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
        f.write("\n")


def load_system_info(path: str | os.PathLike[str]) -> dict:
    """Load system information from a JSON file.

    Parameters
    ----------
    path : str or PathLike
        Input file path.

    Returns
    -------
    dict
        System information.
    """
    with open(path) as f:
        return json.load(f)
