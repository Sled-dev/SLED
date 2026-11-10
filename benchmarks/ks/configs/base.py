import ml_collections

def get_config(seed=42):
    """Kuramoto-Sivashinsky (t in the first 35% of the reference): the phase-1
    configuration (2,000 SOAP steps, uniform sampler).

    Every second-order run of a seed restores the checkpoint trained with this
    config (see configs/batch_size_study.py, configs/sampler_comparison.py)."""
    config = ml_collections.ConfigDict()

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "SLED_KS"
    wandb.name = "phase1_{seed}".format(seed=seed)
    wandb.tag = None

    # Physics-informed initialization
    config.use_pi_init = False

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "gradnorm"
    weighting.init_weights = ml_collections.ConfigDict({"ics": 1000.0, "res": 1.0})
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    # Domain Importance adapation
    config.dia = dia = ml_collections.ConfigDict()
    dia.type = "uniform"
    dia.batch_size = 64*128
    dia.batch_size_eval = None

    arch = config.arch = ml_collections.ConfigDict()
    arch.out_dim = 1
    arch.input_dim = 2
    arch.activation = "tanh"
    arch.periodicity = ml_collections.ConfigDict({"period": (1.0,), "axis": (1,)})
    arch.reparam = None
    arch.arch_name = "Mlp"
    arch.num_layers = 5
    arch.hidden_dim = 64
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 2.0, "embed_dim": 64})


    config.optim = optim = ml_collections.ConfigDict()
    # SOAP for phase 1, broyden.broyden (SSBroyden2 with safeguards) for the
    # quasi-Newton phase; "Soap_SSBroyden2_original" selects broyden.broyden_original
    optim.optimizer = "Soap_SSBroyden2"
    config.optim.first_order_steps = 2_000
    config.optim.second_order_steps = 20_000

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
