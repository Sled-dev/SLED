import ml_collections

import jax.numpy as jnp


def get_config(seed=42):
    """Rayleigh-Taylor (Ra = 1e6), single time window t in [0, 4.75] (jaxpi2's
    time_range (0.1, 0.6), see utils.py).

    Phase 1: Mlp 5 x 64 swish with periodic + Fourier embedding, gradnorm
    weighting, 20,000 SOAP steps on the LAS first-order sampler. LAS is
    load-bearing: with uniform residual points phase 1 ends on the spurious
    quiescent branch (an exact zero-residual solution of the PDE, l2 ~ 0.83);
    20,000 LAS steps hand over at l2 ~ 0.35 - 0.45.
    The residual points are drawn in the mesh's extent (x in [0.0015, 0.9985],
    y in [0.0015, 1.9985]); jaxpi2 uses [0,1] x [0,2]."""
    config = ml_collections.ConfigDict()

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "SLED_RT"
    wandb.name = "phase1_{seed}".format(seed=seed)
    wandb.tag = None

    # Physics-informed initialization
    config.use_pi_init = False

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "gradnorm"
    weighting.init_weights = ml_collections.ConfigDict(
        {"u_ic": 100.0, "v_ic": 100.0, "temp_ic": 10.0, "u_bc": 100.0, "v_bc": 100.0,
         "temp_bc": 10.0, "ru": 1.0, "rv": 1.0, "rc": 1.0, "re": 1.0}   # more weight on temp ic/bc
    )
    weighting.momentum = 0.9
    weighting.update_every_steps = 1_000

    # Domain Importance adaptation (first-order phase)
    config.dia = dia = ml_collections.ConfigDict()
    dia.type = "las"
    dia.batch_size = 64 * 128
    dia.batch_size_eval = None
    dia.beta = 0.2
    dia.tau = 0.002

    arch = config.arch = ml_collections.ConfigDict()
    arch.out_dim = 4          # u, v, p, T
    arch.input_dim = 3        # t, x, y
    arch.activation = "swish"
    arch.periodicity = ml_collections.ConfigDict({"period": (2 * jnp.pi,), "axis": (1,)})   # x in [0, 1]
    arch.reparam = None
    arch.arch_name = "Mlp"
    arch.num_layers = 5
    arch.hidden_dim = 64
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1, "embed_dim": 64})

    config.optim = optim = ml_collections.ConfigDict()
    # SOAP for phase 1, broyden.broyden (SSBroyden2 with safeguards) for the
    # quasi-Newton phase; "Soap_SSBroyden2_original" selects broyden.broyden_original
    optim.optimizer = "Soap_SSBroyden2"
    config.optim.first_order_steps = 20_000   # RT hands over after 20k SOAP (LAS) steps
    config.optim.second_order_steps = 40_000

    optim.beta1 = 0.95
    optim.beta2 = 0.95
    optim.learning_rate = 3e-3
    optim.eps = 1e-8
    optim.decay_rate = 0.9
    optim.schedule_free = False
    optim.decay_steps = 2000
    optim.warmup_steps = 2000

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = None
    saving.num_keep_ckpts = 0

    # Integer for PRNG random seed.
    config.seed = seed

    return config
