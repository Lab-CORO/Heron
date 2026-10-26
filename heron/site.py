"""The site: what differs from one machine to another.

Read from site.yaml at the repository root (or the file HERON_SITE names). It is not
versioned, so personal identifiers (SLURM accounts, paths) never reach git; see
site.example.yaml. Without the file, the defaults reproduce the Narval setup that
existed before it: SLURM, `module load apptainer`, training in capa_unet.sif.
"""

import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from heron.errors import HeronConfigError

ARCHES = ("amd64", "aarch64")
SCHEDULERS = ("slurm", "local")
RUNTIMES = ("venv", "apptainer")
_MACHINE_TO_ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "aarch64", "arm64": "aarch64"}
_KEYS = {"scheduler", "arch", "slurm", "modules", "data_root", "training", "images"}

DEFAULT_IMAGES = {
    "amd64": {"generation": "docker_images/data_generation.sif", "training": "capa_unet.sif"},
    "aarch64": {"generation": None, "training": None},
}


def site_file() -> Path:
    from heron.paths import REPO_ROOT
    return Path(os.environ.get("HERON_SITE", REPO_ROOT / "site.yaml"))


def _raw() -> dict:
    path = site_file()
    if not path.is_file():
        if "HERON_SITE" in os.environ:
            raise HeronConfigError(f"HERON_SITE points to {path}, which does not exist")
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise HeronConfigError(f"{path}: expected a mapping of keys to values")
    unknown = set(data) - _KEYS
    if unknown:
        raise HeronConfigError(f"{path}: unknown key(s) {', '.join(sorted(unknown))} (allowed: {', '.join(sorted(_KEYS))})")
    return data


def configured_data_root() -> Path | None:
    """data_root from the site file, if it sets one (paths.data_root decides precedence)."""
    value = _raw().get("data_root")
    return Path(value).expanduser() if value else None


def host_arch() -> str:
    machine = platform.machine().lower()
    if machine not in _MACHINE_TO_ARCH:
        raise HeronConfigError(f"unsupported machine architecture {machine!r} (supported: {', '.join(ARCHES)})")
    return _MACHINE_TO_ARCH[machine]


@dataclass(frozen=True)
class Site:
    source: Path | None          # the site file read, or None for the defaults
    scheduler: str               # slurm | local
    arch: str                    # amd64 | aarch64
    account_gpu: str | None
    account_cpu: str | None
    modules: dict                # runtime (apptainer, venv) -> modules to load
    training_runtime: str        # venv | apptainer
    venv: Path                   # the Python environment jobs activate in the venv runtime
    images: dict                 # arch -> {generation, training}: path or None

    def image(self, role: str) -> Path:
        """The image for a role (generation, training) on this site's architecture."""
        value = (self.images.get(self.arch) or {}).get(role)
        where = self.source or "the defaults (no site.yaml)"
        if not value:
            raise HeronConfigError(
                f"no {role} image for {self.arch} in {where}. Set images.{self.arch}.{role} in site.yaml "
                f"(an Apptainer .sif meeting docs/images.md)")
        path = Path(value)
        if not path.is_file():
            raise HeronConfigError(f"{role} image for {self.arch} not found: {path} (from {where})")
        return path

    def account(self, gpu: bool) -> str | None:
        return self.account_gpu if gpu else self.account_cpu

    def modules_for(self, runtime: str) -> list[str]:
        return list(self.modules.get(runtime) or [])


def load_site(arch: str | None = None, local: bool = False) -> Site:
    """The site, with command-line overrides: --arch forces the architecture, --local the scheduler."""
    data = _raw()
    source = site_file() if data or site_file().is_file() else None
    where = source or "defaults"

    scheduler = "local" if local else data.get("scheduler", "slurm")
    if scheduler not in SCHEDULERS:
        raise HeronConfigError(f"{where}: scheduler must be one of {', '.join(SCHEDULERS)}, got {scheduler!r}")

    arch = arch or data.get("arch", "auto")
    arch = host_arch() if arch == "auto" else arch
    if arch not in ARCHES:
        raise HeronConfigError(f"{where}: arch must be auto or one of {', '.join(ARCHES)}, got {arch!r}")

    slurm = data.get("slurm") or {}
    if set(slurm) - {"account_gpu", "account_cpu"}:
        raise HeronConfigError(f"{where}: slurm accepts account_gpu and account_cpu")

    modules = {"apptainer": ["apptainer"], "venv": []}
    modules.update(data.get("modules") or {})
    if set(modules) - set(RUNTIMES):
        raise HeronConfigError(f"{where}: modules accepts {', '.join(RUNTIMES)}")

    training = data.get("training") or {}
    if set(training) - {"runtime", "venv"}:
        raise HeronConfigError(f"{where}: training accepts runtime and venv")
    runtime = training.get("runtime", "apptainer")
    if runtime not in RUNTIMES:
        raise HeronConfigError(f"{where}: training.runtime must be one of {', '.join(RUNTIMES)}, got {runtime!r}")
    venv = Path(training.get("venv") or sys.prefix).expanduser()

    images = {a: dict(v) for a, v in DEFAULT_IMAGES.items()}
    for a, roles in (data.get("images") or {}).items():
        if a not in ARCHES or not isinstance(roles, dict) or set(roles) - {"generation", "training"}:
            raise HeronConfigError(f"{where}: images.<{'|'.join(ARCHES)}> accepts generation and training")
        images[a].update(roles)
    # Relative image paths are relative to the data root, resolved now: a later change of
    # data root (heron doctor works in its own) must not move the images.
    from heron.paths import data_root
    root = data_root()
    for roles in images.values():
        for role, value in roles.items():
            if value:
                path = Path(value).expanduser()
                roles[role] = str(path if path.is_absolute() else root / path)

    return Site(source=source, scheduler=scheduler, arch=arch,
                account_gpu=slurm.get("account_gpu"), account_cpu=slurm.get("account_cpu"),
                modules=modules, training_runtime=runtime, venv=venv, images=images)
