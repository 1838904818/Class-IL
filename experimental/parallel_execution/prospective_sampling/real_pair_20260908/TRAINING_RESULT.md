# Common Task-0 encoder training

Completed and audited locally on 8 September 2026. This is new real-data encoder training, not a full OFRA or paired P/O classification experiment.

The seed-1 FT256x4 encoder and temporary multiclass classifier trained for eight epochs on the 11,118 common Task-0 fitting rows. Adam used learning rate 0.001, zero weight decay and batch size 384. The completed state has 232 productive optimizer steps and 88,944 row presentations. The cost ledger includes both the initial step and resumed attempt, with no unknown tail or repeated uncommitted steps.

Observed attempt time totals 53.94 seconds. Peak PyTorch CUDA allocated memory is 3.899 GiB and reserved memory is 4.334 GiB. These are framework measurements, not whole-system GPU memory. Hardware is an RTX 4060 Ti, Python 3.11.9, PyTorch 2.11.0+cu128 and NumPy 2.4.6. This differs from the historical A100 environment.

The first and last batch training losses are 0.69804 and 0.02967. They concern different batches and are not accuracy, validation loss or evidence of improved generalization. No calibration or official-test evaluation occurred. Local journals and checkpoints provide tracking; this run did not log online to W&B.

Terminal CLI status: COMPLETE. Independent post-run verifier status: AUDIT_PASS. Final checkpoint SHA-256: 2d75d0b4dfc85ed20546c0679d23fbd3624064db4486b6cb331be1d40063ea02. The historical PAUSED.json remains an earlier event; COMPLETE.json identifies current completion.

Remaining: export the frozen shared representation, execute matched downstream arms with complete exposure/memory accounting, evaluate locked test data and extend the frozen protocol across paired seeds. No novel-algorithm benefit or ETG intervention utility follows from this encoder stage.
