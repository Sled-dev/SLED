class BaseEvaluator:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.log_dict = {}

    def log_losses(self, params, batch, point_weights):
        losses = self.model.losses(params, batch, point_weights)

        for key, values in losses.items():
            self.log_dict[key + "_loss"] = values

    def log_weights(self, state):
        weights = state.weights
        for key, values in weights.items():
            self.log_dict[key + "_weight"] = values

    def __call__(self, state, batch):
        # Initialize the log dict
        self.log_dict = {}
        params = state.params

        if self.config.logging.log_losses:
            self.log_losses(params, batch, state.point_weights)

        if self.config.logging.log_weights:
            self.log_weights(state)

        if self.config.logging.log_errors:
            self.log_errors(state.params)
            
        return self.log_dict

