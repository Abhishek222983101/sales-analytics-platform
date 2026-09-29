"""Pytest bootstrap: make the project root importable so tests can `import core...`."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))
