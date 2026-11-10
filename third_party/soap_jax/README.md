# soap-jax (vendored)

The SOAP optimizer (Vyas et al., 2024, https://arxiv.org/abs/2409.11321) in JAX,
from https://github.com/haydn-jones/SOAP_JAX at tag v0.1.0 (commit 2c0c34f, MIT
licence, see LICENSE), with the following changes to `src/soap_jax/soap.py`.
This is the exact optimizer that trained every phase-1 hand-over of the study;
a different SOAP version gives different hand-over checkpoints.

- `exp_avg` is tracked in the projected (Kronecker-eigenbasis) space, consistent
  with `exp_avg_sq`, so both moments live in the same basis.
- `exp_avg` is rotated into the new basis whenever the preconditioner `Q` is
  refreshed.
- Bias correction uses `effective_step = count - 1` (step 1 is the
  initialisation step, not a real update); the preconditioner refresh schedule
  counts from the same step (`effective_step % precondition_frequency == 0`
  instead of `count % precondition_frequency == 0`), i.e. it moves by one step.
- Weight decay is applied after the learning-rate scaling (AdamW-style decoupled
  decay), and not on the first step.

The package version is `0.1.0+sled`; `setup.py` of the repository requires
exactly that version, so an install without the vendored copy fails instead of
fetching the unpatched soap-jax. Install with `pip install ./third_party/soap_jax`
(requirements.txt does this).
