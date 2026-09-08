# Pretraining telemetry scope

The optional Python `metric_send` callback receives allowlisted aggregate events, not sample identities, labels, local paths or credentials. The CLI does not initialize W&B. An event contains an opaque attempt identifier and the optimizer step, epoch index, batch size, loss, duration and available CUDA memory measurements.

On an ordinary callback exception, the runner attempts to checkpoint the completed optimizer step and then fails visibly. If that checkpoint succeeds, resuming continues after that step rather than repeating it. Checkpoint failure, process termination and machine failure are not covered by this guarantee. Transport delivery may be uncertain: no automatic resend or exactly-once cloud delivery is claimed. A resumed invocation uses a new attempt ID, and missing cloud points require separate reconciliation against local records.

The metric adapter source hash is bound into input validation and checked at checkpoint and completion boundaries. Resume rejects a changed adapter. The caller-supplied transport implementation is not yet hash-bound; a production W&B launcher must bind its own code and explicit destination before experimental use. Callback latency and failure-checkpoint I/O contribute to total elapsed cost but not the optimizer batch-duration metric.

Local validation on 8 September 2026: all 51 fair-comparison unit tests passed, including synthetic transport failure/resume equality and adapter-drift rejection. These are software tests on tiny synthetic inputs, not real-data training, GPU profiling, cloud-delivery validation, or evidence of classification performance.
