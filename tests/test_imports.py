"""Resolve package modules and all owned-library imports, including local imports."""

import ast
import importlib
from pathlib import Path

import exosims_plugins

OWNERS = {
    "hwoutils",
    "zodi",
    "yippy",
    "optixstuff",
    "skyscapes",
    "orbix",
    "coronagraphoto",
    "jaxedith",
}


def test_import_all():
    """Exact import contract: every module and imported owner name must resolve."""
    package_dir = Path(next(iter(exosims_plugins.__path__)))
    sources = sorted(package_dir.rglob("*.py"))
    for path in sources:
        parts = path.relative_to(package_dir.parent).with_suffix("").parts
        importlib.import_module(".".join(parts).removesuffix(".__init__"))
    for path in [*sources, *Path("tests").rglob("*.py")]:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in OWNERS:
                    module = importlib.import_module(node.module)
                    for alias in node.names:
                        assert hasattr(module, alias.name), (path, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in OWNERS:
                        importlib.import_module(alias.name)
