"""Allen-Cahn with the initial condition of Kiyani et al. (2025):

    u_t - eps u_xx + kappa (u^3 - u) = 0,   eps = 1e-4,  kappa = 5,
    x in [-1, 1] periodic, t in [0, 1],     u(x, 0) = x^2 sin(2 pi x).

The PDE of the classic Allen-Cahn benchmark (Raissi et al., 2019); only the
initial condition differs (there u0 = x^2 cos(pi x), whose seam corner limits
every reference at ~1e-5). The initial condition comes from the dataset (utils.get_dataset) and
kappa from the dataset's PDE-parameter dict. Periodicity is hard-enforced by the
architecture's periodic embedding.
"""
from functools import partial
import jax.numpy as jnp
from jax import jit, grad, vmap
from pipeline.models import PINN
from pipeline.evaluator import BaseEvaluator

EPS = 1e-4


class AllenCahn(PINN):
    def __init__(self, config, axes_dict, initial_dict, pde_params_dict=None):
        super().__init__(config=config, axes_dict=axes_dict, initial_dict=initial_dict,
                         pde_params_dict=pde_params_dict)
        self.kappa = float((pde_params_dict or {}).get("kappa", 5.0))
        self.u0 = initial_dict["u0"]
        self.t_star = axes_dict["t"]
        self.x_star = axes_dict["x"]
        self.t0 = self.t_star[0]
        self.t1 = self.t_star[-1]
        self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0)), (None, 0, None))

    def u_net(self, params, t, x):
        z = jnp.stack([t, x])
        _, u = self.state.apply_fn(params, z)
        return u[0]

    def r_net(self, params, t, x):
        u = self.u_net(params, t, x)
        u_t = grad(self.u_net, argnums=1)(params, t, x)
        u_xx = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x)
        return u_t + self.kappa * (u**3 - u) - EPS * u_xx

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test):
        u_pred = self.u_pred_fn(params, self.t_star, self.x_star)
        return jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)

    @partial(jit, static_argnums=(0,))
    def r_losses_pp(self, params, batch, weights):
        r_pred = vmap(self.r_net, in_axes=(None, 0, 0))(params, batch[:, 0], batch[:, 1])
        return r_pred**2 * weights["res"]

    @partial(jit, static_argnums=(0,))
    def loss(self, params, weights, batch, point_weights):
        return (weights["ics"] * self.loss_ics(params, batch, point_weights)
                + weights["res"] * self.loss_res(params, batch, point_weights))

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch, point_weights):
        return {"ics": self.loss_ics(params, batch, point_weights),
                "res": self.loss_res(params, batch, point_weights)}

    @partial(jit, static_argnums=(0,))
    def pred_last_t(self, params):
        return {"u0": vmap(self.u_net, (None, None, 0))(params, self.t1, self.x_star)}

    @partial(jit, static_argnums=(0,))
    def loss_ics(self, params, batch, point_weights):
        u_pred = vmap(self.u_net, (None, None, 0))(params, self.t0, self.x_star)
        return jnp.mean((self.u0 - u_pred) ** 2)

    @partial(jit, static_argnums=(0,))
    def loss_res(self, params, batch, point_weights):
        r_pred = vmap(self.r_net, (None, 0, 0))(params, batch[:, 0], batch[:, 1])
        return jnp.mean((r_pred * point_weights) ** 2)

    @property
    def loss_fns(self):
        return {"ics": self.loss_ics, "res": self.loss_res}


class AllenCahnEvaluator(BaseEvaluator):
    def __init__(self, config, model, sol_dict):
        super().__init__(config, model)
        self.u_ref = sol_dict["u"]

    def log_errors(self, params):
        self.log_dict["l2_error"] = self.model.compute_l2_error(params, self.u_ref)
