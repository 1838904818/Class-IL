from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ncd.plot_exp2_four_cell_roc import parse_roc_svg


class NcdPlotExp2FourCellRocTest(unittest.TestCase):
    def test_parses_and_normalizes_registered_series(self) -> None:
        paths = []
        for label, color in (
            ("B1", "#355C7D"),
            ("B2", "#D4A017"),
            ("B3", "#C96A3D"),
            ("B4c", "#7A8F3A"),
            ("B4k", "#B04A76"),
            ("NCD", "#2676A8"),
        ):
            paths.append(
                f'<path d="M 92 526 L 92 96 L 652 96" fill="none" stroke="{color}"/>'
            )
        svg = '<svg xmlns="http://www.w3.org/2000/svg">' + "".join(paths) + "</svg>"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "roc.svg"
            path.write_text(svg, encoding="utf-8")
            curves = parse_roc_svg(path)
        self.assertEqual(len(curves), 6)
        self.assertEqual(curves["NCD"], [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)])


if __name__ == "__main__":
    unittest.main()
