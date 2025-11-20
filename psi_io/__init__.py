from .psi_io import *

try:
    from importlib.metadata import version as _pkg_version
    from importlib.metadata import PackageNotFoundError
    from pathlib import Path
    __version__ = _pkg_version("psi-io")  # type: ignore[assignment]
except PackageNotFoundError as e:  # dev/editable without metadata
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # pip install tomli

    pyproject = Path(__file__).parents[1].resolve() / 'pyproject.toml'
    data = tomllib.loads(pyproject.read_text())

    project_version = data.get("project", {}).get("version", "0+unknown")
    project_version = project_version.replace('"', '').replace("'", '')
    __version__ = project_version
