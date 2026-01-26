# noxfile.py
from __future__ import annotations

import os
import re
import json
import platform
import subprocess
from pathlib import Path
import nox

nox.options.reuse_existing_virtualenvs = True
pyproject = nox.project.load_toml()

PY_VERSIONS = nox.project.python_versions(pyproject)
try:
    txt = Path('.python-version').read_text(encoding="utf-8").strip().splitlines()[0]
    m = re.search(r"(\d+)\.(\d+)", txt)  # grab major.minor; ignore patch/suffix
    SYS_PYTHON = f"{m.group(1)}.{m.group(2)}" if m else PY_VERSIONS[-1]
except FileNotFoundError as e:
    SYS_PYTHON = PY_VERSIONS[-1]

PROJECT_NAME = pyproject["project"]["name"]
PROJECT_NAME_PATH = Path(__file__).parent.resolve()
_ARTIFACTS = PROJECT_NAME_PATH / ".nox" / "_artifacts"

WHEEL_DIR = _ARTIFACTS / "wheels"
SDIST_DIR = _ARTIFACTS / "sdist"
WHEELHOUSE_DIR = _ARTIFACTS / "wheelhouse"
DOCDIST_DIR = _ARTIFACTS / "docs"

REPAIR_TOOLS: dict[str, list[str]] = {
    "linux": ["auditwheel"],
    "darwin": ["delocate"],
    "windows": ["delvewheel"],
}

SDIST_DIR.mkdir(parents=True, exist_ok=True)
WHEEL_DIR.mkdir(parents=True, exist_ok=True)
WHEELHOUSE_DIR.mkdir(parents=True, exist_ok=True)
DOCDIST_DIR.mkdir(parents=True, exist_ok=True)


def _darwin_sdk_env() -> dict[str, str]:
    """macOS: provide SDK + baseline target so the Fortran probe succeeds."""
    if platform.system() != "Darwin":
        return {}
    # Prefer already-set values; otherwise best-effort defaults
    env = {}
    env.setdefault("MACOSX_DEPLOYMENT_TARGET", os.environ.get("MACOSX_DEPLOYMENT_TARGET", "11.0"))
    try:
        sdk = subprocess.check_output(
            ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
            text=True
        ).strip()
        env["SDKROOT"] = os.environ.get("SDKROOT", sdk)
        # Help the linkers/compilers see the SDK explicitly:
        env.setdefault("CFLAGS",    f"-isysroot {sdk} -mmacosx-version-min={env['MACOSX_DEPLOYMENT_TARGET']}")
        env.setdefault("CXXFLAGS",  f"-isysroot {sdk} -mmacosx-version-min={env['MACOSX_DEPLOYMENT_TARGET']}")
        env.setdefault("FCFLAGS",   f"-isysroot {sdk} -mmacosx-version-min={env['MACOSX_DEPLOYMENT_TARGET']}")
        env.setdefault("LDFLAGS",   f"-Wl,-syslibroot,{sdk} -mmacosx-version-min={env['MACOSX_DEPLOYMENT_TARGET']}")
    except Exception:
        pass
    return env


def _build_env(session: nox.Session) -> Path:
    """Build a wheel into ./dist and return its path."""
    session.conda_install(
        *pyproject["tool"][PROJECT_NAME].get("conda", []),
        channel="conda-forge"
    )
    session.env.update(_darwin_sdk_env())


def _dist_env(session: nox.Session) -> Path:
    """Environment for installing from built wheels."""
    session.install(
        *pyproject["project"].get("dependencies", []),
    )

    session.run(
        "python", "-m", "pip", "install",
        "--no-index", f"--find-links={WHEELHOUSE_DIR}",
        "--only-binary=:all:", "--report", "-",
        PROJECT_NAME,
    )


@nox.session(venv_backend='conda|mamba|micromamba', python=PY_VERSIONS)
def build(session: nox.Session) -> None:
    """Build the package wheel (with compilers)."""
    _build_env(session)
    session.conda_install(
        *pyproject["build-system"].get("requires", []),
        *pyproject["project"].get("optional-dependencies", {}).get("build", []),
        channel="conda-forge",
    )
    session.run(
        "python", "-m", "build",
        "--wheel", "--outdir", WHEEL_DIR.as_posix(),
        external=False
    )


@nox.session(python=SYS_PYTHON)
def repair(session: nox.Session) -> None:
    """Repair wheels in dist/ into wheelhouse/ using the OS-specific tool."""
    platform_id = platform.system().lower()
    wheels = sorted(WHEEL_DIR.glob("*.whl"))

    match platform_id:
        case "linux":
            session.install("auditwheel")
            for whl in wheels:
                session.run("auditwheel", "show", str(whl))
                session.run("auditwheel", "repair", "-w", str(WHEELHOUSE_DIR), str(whl))
        case "darwin":
            session.install("delocate")
            for whl in wheels:
                session.run("delocate-listdeps", str(whl))
                session.run("delocate-wheel", "-w", str(WHEELHOUSE_DIR), str(whl))
        case "windows":
            session.install("delvewheel")
            for whl in wheels:
                session.run("python", "-m", "delvewheel", "show", str(whl))
                session.run("python", "-m", "delvewheel", "repair", "-w", str(WHEELHOUSE_DIR), str(whl))


@nox.session(python=PY_VERSIONS)
def test(session: nox.Session) -> None:
    """Build the wheel (with compilers), install it, then run pytest from a temp dir."""
    # Build wheel
    _dist_env(session)
    session.install(*pyproject["project"].get("optional-dependencies", {}).get("test", []))

    tmp = session.create_tmp()
    session.chdir(tmp)

    # Pytest
    session.run("pytest", PROJECT_NAME_PATH.as_posix(), *session.posargs)


@nox.session(venv_backend='conda|mamba|micromamba', python=SYS_PYTHON)
def sdist(session: nox.Session) -> None:
    """Build the package wheel (with compilers)."""
    _build_env(session)
    session.conda_install(
        *pyproject["build-system"].get("requires", []),
        *pyproject["project"].get("optional-dependencies", {}).get("build", []),
        channel="conda-forge",
    )
    session.run(
        "python", "-m", "build",
        "--sdist", "--outdir", SDIST_DIR.as_posix(),
        external=False
    )


@nox.session(python=SYS_PYTHON)
def types(session: nox.Session) -> None:
    """Mypy type checking (analyzes source tree).

    .. note::
       Target version can be set via ``nox -s types -- --python-version 3.10``
    """
    session.install(*pyproject["project"].get("optional-dependencies", {}).get("types", []))

    session.run("mypy", *session.posargs)


@nox.session(python=SYS_PYTHON)
def lint(session: nox.Session) -> None:
    """Ruff lint + format check.

    .. note::
       Target version can be set via ``nox -s lint -- --target-version py310``

    .. note::
       Strict mode can be enabled via ``nox -s lint -- --select ALL``
    """
    session.install(*pyproject["project"].get("optional-dependencies", {}).get("lint", []))

    session.run("ruff", "check", PROJECT_NAME, *session.posargs)


@nox.session(python=SYS_PYTHON)
def docs(session: nox.Session) -> None:
    """Build Sphinx docs against the installed wheel."""
    _dist_env(session)
    session.install(*pyproject["project"].get("optional-dependencies", {}).get("docs", []))
    args = pyproject["tool"].get("sphinx_build", {}).get("addopts", [])

    stamp = session.env_dir / ".mapflpy_tag.json"
    session.run(
        "python", "-c",
        (
            "import json; "
            "from importlib import metadata as md; "
            "d=md.distribution('mapflpy'); "
            "txt=d.read_text('WHEEL'); "
            "tags=[]; "
            "tags=[ln.split(':',1)[1].strip() for ln in txt.splitlines() if ln.startswith('Tag: ')]; "
            "out={'tags': tags}; "
            f"open('{stamp}','w').write(json.dumps(out))"
        )
    )
    tag = json.loads(Path(stamp).read_text())

    out_dir = DOCDIST_DIR / f"html-{tag.get('tags', ['none'])[0]}"
    src_dir = PROJECT_NAME_PATH / "docs" / "source"
    session.run("sphinx-build", src_dir.as_posix(), out_dir.as_posix(), *args)


@nox.session(python=SYS_PYTHON)
def build_matrix(session: nox.Session) -> None:
    """Build, repair, and test in order (single entrypoint)."""
    session.notify("sdist")
    session.notify("build")
    session.notify("repair")
    session.notify("test")


@nox.session(python=SYS_PYTHON)
def build_target(session: nox.Session) -> None:
    """Build, repair, and test in order (single entrypoint)."""
    session.notify("sdist")
    session.notify(f"build-{session.python}")
    session.notify("repair")
    session.notify(f"test-{session.python}")


@nox.session(python=SYS_PYTHON)
def build_docs(session: nox.Session) -> None:
    """Build, repair, and test in order (single entrypoint)."""
    session.notify("sdist")
    session.notify(f"build-{session.python}")
    session.notify("repair")
    session.notify(f"test-{session.python}")
    session.notify("docs")


@nox.session(python=SYS_PYTHON)
def build_qa(session: nox.Session) -> None:
    """Build, repair, and test in order (single entrypoint)."""
    session.notify("sdist")
    session.notify(f"build-{session.python}")
    session.notify("repair")
    session.notify(f"test-{session.python}")
    session.notify(f"types")
    session.notify(f"lint")


if __name__ == "__main__":
    nox.main()
