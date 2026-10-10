from __future__ import annotations

import unittest

from ncd.plot_ncd_method_architecture import EDGES, NODES, validate_spec


class NcdPlotMethodArchitectureTest(unittest.TestCase):
    def test_locked_research_boundary_and_required_controls_are_present(self) -> None:
        validate_spec()
        self.assertIn("rq3", NODES)
        self.assertIn(("explainers", "baselines"), EDGES)
        text = " ".join(value[4] + " " + value[5] for value in NODES.values())
        self.assertIn("B1", text)
        self.assertIn("B2", text)
        self.assertIn("not validated", text)


if __name__ == "__main__":
    unittest.main()
