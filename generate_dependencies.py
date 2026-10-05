"""Synchronize Python imports with pyproject.toml dependencies."""

from __future__ import annotations  # noqa:I001

import argparse
import ast
import importlib.metadata
import keyword
import re
import sys
import tomllib
from pathlib import Path

IGNORED_DIRS = {
    ".git",
    ".github",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "site-packages",
}
IMPORT_TO_DISTRIBUTION = {
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
}
REQUIREMENT_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze Python imports and synchronize pyproject.toml."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root directory.")
    parser.add_argument("--include-tests", action="store_true", help="Include imports from tests/.")
    parser.add_argument(
        "--check", action="store_true", help="Only validate pyproject.toml dependencies."
    )
    parser.add_argument("--verbose", action="store_true", help="Print discovered imports.")
    return parser.parse_args()


def normalize_distribution_name(name: str) -> str:
    """Normalize a package distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def extract_requirement_name(requirement: str) -> str | None:
    """Extract a package name from a requirement string."""
    match = REQUIREMENT_NAME_RE.match(requirement)
    return match.group(1) if match else None


def load_project_dependencies(pyproject: Path) -> list[str]:
    """Load runtime dependencies from [project].dependencies."""
    with pyproject.open("rb") as fp:
        data = tomllib.load(fp)
    return list(data.get("project", {}).get("dependencies", []))


def find_python_files(root: Path, include_tests: bool) -> list[Path]:
    """Find Python files under the project root."""
    files = []
    for path in root.rglob("*.py"):
        parts = path.relative_to(root).parts
        if any(part in IGNORED_DIRS for part in parts):
            continue
        if not include_tests and "tests" in parts:
            continue
        files.append(path)
    return sorted(files)


def extract_imports(path: Path) -> set[str]:
    """Extract top-level external import names from a Python file."""
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.add(node.module.split(".")[0])
    return imports


def find_local_modules(root: Path) -> set[str]:
    """Find local module and package names."""
    modules = set()
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if any(part in IGNORED_DIRS for part in relative.parts):
            continue
        modules.add(
            relative.parts[-2]
            if path.name == "__init__.py" and len(relative.parts) > 1
            else path.stem
        )
    return modules


def build_import_distribution_map() -> dict[str, set[str]]:
    """Map import names to installed distribution names."""
    mapping: dict[str, set[str]] = {}
    for import_name, distributions in importlib.metadata.packages_distributions().items():
        mapping.setdefault(import_name, set()).update(distributions)
    return mapping


def resolve_distribution(import_name: str, mapping: dict[str, set[str]]) -> str | None:
    """Resolve an import name to a distribution name."""
    if import_name in IMPORT_TO_DISTRIBUTION:
        return IMPORT_TO_DISTRIBUTION[import_name]
    distributions = mapping.get(import_name)
    return sorted(distributions)[0] if distributions else None


def get_installed_version(distribution: str) -> str | None:
    """Get an installed distribution version."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def update_pyproject_dependencies(pyproject: Path, missing: dict[str, str]) -> list[str]:
    """Append missing dependencies to [project].dependencies."""
    source = pyproject.read_text(encoding="utf-8")
    project_match = re.search(r"(?m)^\[project\]\s*$", source)
    if not project_match:
        raise ValueError("pyproject.toml does not contain a [project] table.")
    project_start = project_match.end()
    next_table = re.search(r"(?m)^\[[^\]]+\]\s*$", source[project_start:])
    project_end = project_start + next_table.start() if next_table else len(source)
    project_text = source[project_start:project_end]
    dependencies_match = re.search(r"(?ms)^dependencies\s*=\s*\[(.*?)^\s*\]", project_text)
    if not dependencies_match:
        raise ValueError("[project].dependencies must exist as an array in pyproject.toml.")
    additions = [missing[name] for name in sorted(missing)]
    insertion = "".join(f'    "{requirement}",\n' for requirement in additions)
    body_end = dependencies_match.end(1)
    new_project_text = project_text[:body_end] + insertion + project_text[body_end:]
    pyproject.write_text(
        source[:project_start] + new_project_text + source[project_end:], encoding="utf-8"
    )
    return additions


def main() -> int:
    """Analyze imports and synchronize pyproject.toml."""
    args = parse_args()
    root = args.root.resolve()
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        print(f"ERROR: pyproject.toml not found: {pyproject}", file=sys.stderr)
        return 1
    declared = load_project_dependencies(pyproject)
    declared_names = {
        normalize_distribution_name(name)
        for name in (extract_requirement_name(req) for req in declared)
        if name
    }
    python_files = find_python_files(root, args.include_tests)
    print(f"Project root: {root}")
    print(f"Python files: {len(python_files)}")
    imports = set()
    for path in python_files:
        file_imports = extract_imports(path)
        imports.update(file_imports)
        if args.verbose and file_imports:
            print(f"\n{path.relative_to(root)}")
            for name in sorted(file_imports):
                print(f"  {name}")
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    local_modules = find_local_modules(root)
    external = {
        name
        for name in imports
        if name not in stdlib and name not in local_modules and not keyword.iskeyword(name)
    }
    mapping = build_import_distribution_map()
    discovered: dict[str, str] = {}
    unresolved = set()
    for import_name in sorted(external):
        distribution = resolve_distribution(import_name, mapping)
        if not distribution:
            unresolved.add(import_name)
            continue
        version = get_installed_version(distribution)
        if version is None:
            unresolved.add(import_name)
            continue
        discovered[normalize_distribution_name(distribution)] = f"{distribution}=={version}"
    if unresolved:
        print("\nWARNING: Could not resolve imports:")
        for name in sorted(unresolved):
            print(f"  {name}")
    missing = {name: req for name, req in discovered.items() if name not in declared_names}
    if not missing:
        print("\npyproject.toml is synchronized with discovered imports.")
        return 0
    print("\nMissing dependencies:")
    for req in missing.values():
        print(f"  {req}")
    if args.check:
        print("\nERROR: pyproject.toml is missing discovered dependencies.", file=sys.stderr)
        print("Run: python generate_dependencies.py", file=sys.stderr)
        return 1
    additions = update_pyproject_dependencies(pyproject, missing)
    print(f"\nUpdated: {pyproject}")
    print(f"Added dependencies: {len(additions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
