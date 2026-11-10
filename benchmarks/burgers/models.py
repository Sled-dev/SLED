from functools import partial

import jax.numpy as jnp
from jax import jit, grad, vmap, jvp, value_and_grad
from pipeline.models import PINN
from pipeline.evaluator import BaseEvaluator


class Burgers(PINN):
    def __init__(self, config, axes_dict, initial_dict,  pde_params_dict=None):
        super().__init__(config, initial_dict, axes_dict, pde_params_dict= pde_params_dict)

        self.u0 = initial_dict["u0"]
        self.t_star = axes_dict["t"]
        self.x_star = axes_dict["x"]

        self.t0 = self.t_star[0]
        self.t1 = self.t_star[-1]

        # Predictions over a grid
        self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0)), (None, 0, None))
        self.r_pred_fn = vmap(vmap(self.r_net, (None, None, 0)), (None, 0, None))

    def u_net(self, params, t, x):
        z = jnp.stack([t, x])
        _, u = self.state.apply_fn(params, z)
        return u[0]

    def r_net(self, params, t, x):
        z = jnp.stack([t, x])
        u_z = lambda z_: self.u_net(params, z_[0], z_[1])

        u, u_grad = value_and_grad(u_z)(z)
        _, u_hess_x = jvp(
            grad(u_z),
            (z,),
            (jnp.array([0.0, 1.0], dtype=z.dtype),),
        )
        return u_grad[0] + u * u_grad[1] - 0.01 / jnp.pi * u_hess_x[1]
    
    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch, point_weights):
        # Initial condition loss
        ics_loss = self.loss_ics(params, batch, point_weights)
        res_loss = self.loss_res(params, batch, point_weights)
        return {"ics": ics_loss, "res": res_loss}

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test):
        u_pred = self.u_pred_fn(params, self.t_star, self.x_star)
        error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        return error
    
    @partial(jit, static_argnums=(0,))
    def pred_last_t(self, params):
        u_pred = self.u_pred_fn(params, self.t_star[-1:], self.x_star)
        return {"u0": u_pred[0]}
    
    @partial(jit, static_argnums=(0,))
    def loss(self, params, weights, batch, point_weights):
        ics_loss = self.loss_ics(params, batch, point_weights)
        res_loss = self.loss_res(params, batch, point_weights)
        return weights["ics"] * ics_loss + weights["res"] * res_loss
    
    @partial(jit, static_argnums=(0,))
    def r_losses_pp(self, params, candidate_batch, weights):
        t = candidate_batch[:, 0]
        x = candidate_batch[:, 1]
        r_pred = vmap(self.r_net, in_axes=(None, 0, 0))(params, t, x)
        return r_pred**2 * weights["res"]
    
    @partial(jit, static_argnums=(0,))
    def pred_domain(self, params):
        u_pred = self.u_pred_fn(params, self.t_star, self.x_star)
        return {"u": u_pred}

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
        return {
            "ics": self.loss_ics,
            "res": self.loss_res,
        }

class BurgersEvaluator(BaseEvaluator):
    def __init__(self, config, model, sol_dict):
        super().__init__(config, model)
        self.u_ref = sol_dict["u"]

    def log_errors(self, params):
        l2_error = self.model.compute_l2_error(params, self.u_ref)
        self.log_dict["l2_error"] = l2_error
