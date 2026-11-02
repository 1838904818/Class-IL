# CSE Freq/WACB raw completion

Acceptance is pending for raw deliveries; completion is not independent scientific acceptance or a passed S1 test. No training, weights, thresholds, confusion matrices, or completed stages were changed for this delivery. Original result hashes and source-relative locations are embedded. Per-stage default and 1/2/5% operating-point matrices and original calibration evidence are preserved. Versions are from the installed pinned environments, not invented per-cell telemetry. Small JSON delivery does not replace protected checkpoints/logs or their later archival checks.

NSL-KDD and Freq/WACB additions are unregistered extensions. Do not pool them into the original registered family or claim overall research completion. RQ1, RQ3, substantive innovation, manuscript QA, and backup reconciliation remain open.

All ten cells: Linux job493534, sklearn1.7.2 overlay, torch2.6.0+cu118. CSE per_class.py routes these methods to section165_confirmation/cse-cic-ids2018/<method>_seedN/result.json; this preserves that existing consumer route.

Windows/Linux provenance: CSE iCaRL10 and BiC10-12 were computed locally on Windows. iCaRL11-14 and every B cell were computed on Linux. BiC13 imported the exact completed Windows task0-2 weights/progress, trained task3 on Linux, and evaluated all four stages on Linux. BiC14 is still training and is deliberately absent.

No completed seed was trained twice. Windows diagnostic reevaluation of Linux-trained checkpoints is not a second training result. Linux verification493605 reproduced all CSE iCaRL11-14 and NSL DER++10-14 TEST matrices and holdout counts exactly. Windows reevaluation differed at float32 decision boundaries: NSL DER++11 final1% has one additional Normal-to-Probe error, giving Windows-minus-Linux final BAcc = -100/(9711*5) = -0.002059520131809288 percentage points; the other four DER++ final1% matrices matched. For CSE iCaRL11, three calibrated stage matrices differed and default matrices matched; its exact final1% BAcc delta has not yet been quantified in the retained diagnostic output and must not be guessed. The original Linux matrices, not Windows diagnostic values, are delivered for Linux-trained cells.

S1 must not silently mix runtimes. Where both evaluation versions exist, use Linux evaluation for every seed of that baseline, retaining the Windows diagnostic provenance. A Linux-harmonized evaluation of locally completed seed10 (and BiC10-12) is pending, without retraining those seeds. These compatibility files alone do not assert a runtime-harmonized S1 comparison.
