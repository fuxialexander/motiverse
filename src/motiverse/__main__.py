"""Entry point for running the genome motif analysis module as a script.

This module enables running the CLI via:
    python -m motiverse

This avoids import conflicts that can occur when importing main() in __init__.py.
"""

from .main import main

if __name__ == "__main__":
    main()
