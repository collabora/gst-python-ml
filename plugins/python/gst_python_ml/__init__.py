import sys
from pathlib import Path

# the wheel nests the element modules here, a checkout puts them on sys.path already
INSTALLED_ELEMENT_MODULES = Path(__file__).parent / "plugins" / "python"
if INSTALLED_ELEMENT_MODULES.is_dir():
    sys.path.insert(0, str(INSTALLED_ELEMENT_MODULES))


def launch():
    import pyml_launch

    pyml_launch.main()


def mcp():
    import pyml_mcp

    pyml_mcp.main()
