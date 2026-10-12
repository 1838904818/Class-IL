# Section 14 local gate cross-check

- Source commit: `1224cb677daeb4445587d173867e000316e8db91`
- Device: NVIDIA GeForce RTX 4060 Ti (`cuda`)
- Scope: NSL-KDD and UNSW-NB15; seeds 1 and 2; 48 raw rows.
- Decision: `STOP_AND_REPAIR_BENCHMARK`.
- Evidence boundary: this is a local deterministic cross-check while DICC job
  `485119` is queued. It does not replace the protected DICC gate receipt and
  cannot authorize the five-seed matrix.

