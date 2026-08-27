"""
scripts.common.show_tree: A simple script to show the structure of Python files
in a directory or a single Python file.

How to use:
    python show_tree.py <path>
"""

import ast
import sys
from pathlib import Path


if len(sys.argv) < 2:
    print("Usage: python show_tree.py <path>")
    sys.exit(1)

root = Path(sys.argv[1])

if not root.exists():
    print(f"Path does not exist: {root}")
    sys.exit(1)

if root.is_file():
    if root.suffix != ".py":
        print(f"Expected a Python file (.py), but got: {root}")
        sys.exit(1)

    pyfiles = [root]

else:
    pyfiles = root.rglob("*.py")


for pyfile in pyfiles:
    print(f"├── {pyfile}")

    try:
        tree = ast.parse(pyfile.read_text(encoding="utf-8"))

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                print(f"│   ├── def {node.name}()")

            elif isinstance(node, ast.AsyncFunctionDef):
                print(f"│   ├── async def {node.name}()")

            elif isinstance(node, ast.ClassDef):
                print(f"│   ├── class {node.name}")

                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        print(f"│   │   ├── def {item.name}()")

                    elif isinstance(item, ast.AsyncFunctionDef):
                        print(f"│   │   ├── async def {item.name}()")

    except Exception as e:
        print(f"│   └── Error parsing file: {e}")