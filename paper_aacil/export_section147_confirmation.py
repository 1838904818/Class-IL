"""Export only completed selected confirmation cells bound to the selection commit."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_section138_screen import main

if __name__ == "__main__":
    main(confirmation=True)
