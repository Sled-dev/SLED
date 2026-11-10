from functools import partial

import jax.numpy as jnp
from jax import jit, jvp, vmap, jacfwd

from pipeline.models import PINN
from pipeline.evaluator import BaseEvaluator





class GinzburgLandau(PINN):
    # residual families in the order r_net returns them (loss-weight keys)
    residual_families = ("ru", "rv")

    def __init__(self, config,  axes_dict, initial_dict, pde_params_dict):
        super().__init__(config=config, axes_dict=axes_dict, initial_dict=initial_dict, pde_params_dict=pde_params_dict)

        self.u0 = initial_dict["u0"]
        self.v0 = initial_dict["v0"]
        self.t_star = axes_dict["t"]
        self.x_star = axes_dict["x"]
        self.y_star = axes_dict["y"]

        # PDE parameters
        self.eps = pde_params_dict["eps"]
        self.k = pde_params_dict["k"]

        # Predictions over a grid
        self.ic_pred_fn = vmap(
            vmap(self.nn_pred_vec, (None, None, None, 0)), (None, None, 0, None)
        )

        self.u_pred_fn = vmap(
            vmap(vmap(self.u_net, (None, None, None, 0)), (None, None, 0, None)),
            (None, 0, None, None),
        )
        self.v_pred_fn = vmap(
            vmap(vmap(self.v_net, (None, None, None, 0)), (None, None, 0, None)),
            (None, 0, None, None),
        )

        self.r_pred_fn = vmap(self.r_net, (None, 0, 0, 0))
        self.domain_pred_fn = vmap(
            vmap(vmap(self.nn_pred_vec, (None, None, None, 0)), (None, None, 0, None)),
            (None, 0, None, None),
        )
        

    def _scaled_inputs(self, t, x, y):
        return jnp.stack([t / self.t_star[-1], x, y])

    def nn_pred_vec(self, params, t, x, y):
        z = self._scaled_inputs(t, x, y)
        _, outputs = self.state.apply_fn(params, z)
        return outputs  # shape (4,)

    def u_net(self, params, t, x, y):
        return self.nn_pred_vec(params, t, x, y)[0]

    def v_net(self, params, t, x, y):
        return self.nn_pred_vec(params, t, x, y)[1]

    def r_net(self, params, t, x, y):
        z = jnp.stack([t, x, y])
        f = lambda z_: self.nn_pred_vec(params, z_[0], z_[1], z_[2])

        uv = f(z)
        u, v = uv

        # Time derivatives only — cheaper than full Jacobian
        e_t = jnp.array([1.0, 0.0, 0.0], dtype=z.dtype)
        _, d_t = jvp(f, (z,), (e_t,))
        u_t, v_t = d_t

        # Spatial Laplacian via two forward-over-forward passes
        e_x = jnp.array([0.0, 1.0, 0.0], dtype=z.dtype)
        e_y = jnp.array([0.0, 0.0, 1.0], dtype=z.dtype)

        _, d2_x = jvp(lambda z_: jvp(f, (z_,), (e_x,))[1], (z,), (e_x,))
        _, d2_y = jvp(lambda z_: jvp(f, (z_,), (e_y,))[1], (z,), (e_y,))

        u_lap = d2_x[0] + d2_y[0]
        v_lap = d2_x[1] + d2_y[1]

        ru = u_t - self.eps * u_lap - self.k * (u - u * (u**2 + v**2) + 1.5 * v * (u**2 + v**2))
        rv = v_t - self.eps * v_lap - self.k * (v - v * (u**2 + v**2) - 1.5 * u * (u**2 + v**2))
        return ru, rv

    def ru_net(self, params, t, x, y):
        u, v = self.nn_pred_vec(params, t, x, y)
        u_t = jacfwd(self.u_net, argnums=1)(params, t, x, y)
        u_xx = jacfwd(jacfwd(self.u_net, argnums=2), argnums=2)(params, t, x, y)
        u_yy = jacfwd(jacfwd(self.u_net, argnums=3), argnums=3)(params, t, x, y)
        u_lap = u_xx + u_yy
        ru = (
            u_t - self.eps * u_lap
            - self.k * (u - u * (u**2 + v**2) + 1.5 * v * (u**2 + v**2))
        )
        return ru

    def rv_net(self, params, t, x, y):
        u, v = self.nn_pred_vec(params, t, x, y)
        v_t = jacfwd(self.v_net, argnums=1)(params, t, x, y)
        v_xx = jacfwd(jacfwd(self.v_net, argnums=2), argnums=2)(params, t, x, y)
        v_yy = jacfwd(jacfwd(self.v_net, argnums=3), argnums=3)(params, t, x, y)
        v_lap = v_xx + v_yy
        rv = (
            v_t - self.eps * v_lap
            - self.k * (v - v * (u**2 + v**2) - 1.5 * u * (u**2 + v**2))
        )
        return rv

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch, point_weights):
        ic_pred = self.ic_pred_fn(params, 0.0, self.x_star, self.y_star)
        u_ic_loss = jnp.mean((ic_pred[..., 0] - self.u0) ** 2)
        v_ic_loss = jnp.mean((ic_pred[..., 1] - self.v0) ** 2)

        ru_pred, rv_pred = self.r_pred_fn(params, batch[:, 0], batch[:, 1], batch[:, 2])
        ru_loss = jnp.mean((ru_pred*point_weights) ** 2)
        rv_loss = jnp.mean((rv_pred*point_weights) ** 2)

        loss_dict = {
            "u_ic": u_ic_loss,
            "v_ic": v_ic_loss,
            "ru": ru_loss,
            "rv": rv_loss,
        }
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_ref, v_ref):
        pred = self.domain_pred_fn(params, self.t_star, self.x_star, self.y_star)
        u_error = jnp.linalg.norm(pred[..., 0] - u_ref) / jnp.linalg.norm(u_ref)
        v_error = jnp.linalg.norm(pred[..., 1] - v_ref) / jnp.linalg.norm(v_ref)

        return u_error, v_error
    
    @partial(jit, static_argnums=(0,))
    def r_losses_pp(self, params, candidate_batch, weights):
        ru_pred, rv_pred = self.r_pred_fn(params, candidate_batch[:, 0], candidate_batch[:, 1], candidate_batch[:, 2])
        return ru_pred**2 * weights["ru"] + rv_pred**2 * weights["rv"]
    
    @partial(jit, static_argnums=(0,))
    def loss(self, params, weights, batch, point_weights):
        ic_pred = self.ic_pred_fn(params, 0.0, self.x_star, self.y_star)
        u_ic_loss = jnp.mean((ic_pred[..., 0] - self.u0) ** 2)
        v_ic_loss = jnp.mean((ic_pred[..., 1] - self.v0) ** 2)

        ru_pred, rv_pred = self.r_pred_fn(
            params, batch[:, 0], batch[:, 1], batch[:, 2]
        )
        # Compute loss
        ru_loss = jnp.mean((ru_pred*point_weights) ** 2)
        rv_loss = jnp.mean((rv_pred*point_weights) ** 2)

        return (u_ic_loss*weights["u_ic"] 
                + v_ic_loss*weights["v_ic"] 
                + ru_loss*weights["ru"] 
                + rv_loss*weights["rv"])
    
    @partial(jit, static_argnums=(0,))
    def pred_last_t(self, params):
        pred = self.ic_pred_fn(params, self.t_star[-1], self.x_star, self.y_star)
        return {"u0": pred[..., 0], 
                "v0": pred[..., 1]}
    
    @partial(jit, static_argnums=(0,))
    def pred_domain(self, params):
        pred = self.domain_pred_fn(params, self.t_star, self.x_star, self.y_star)
        return {"u": pred[..., 0], "v": pred[..., 1]}
    
    @partial(jit, static_argnums=(0,))
    def loss_u_ic(self, params, batch, point_weights):
        pred = self.ic_pred_fn(params, 0.0, self.x_star, self.y_star)
        return jnp.mean((pred[..., 0] - self.u0) ** 2)

    @partial(jit, static_argnums=(0,))
    def loss_v_ic(self, params, batch, point_weights):
        pred = self.ic_pred_fn(params, 0.0, self.x_star, self.y_star)
        return jnp.mean((pred[..., 1] - self.v0) ** 2)

    @partial(jit, static_argnums=(0,))
    def loss_ru(self, params, batch, point_weights):
        ru_pred = vmap(self.ru_net, in_axes=(None, 0, 0, 0))(params, batch[:, 0], batch[:, 1], batch[:, 2])
        return jnp.mean( (ru_pred*point_weights) ** 2)

    @partial(jit, static_argnums=(0,))
    def loss_rv(self, params, batch, point_weights):
        rv_pred = vmap(self.rv_net, in_axes=(None, 0, 0, 0))(params, batch[:, 0], batch[:, 1], batch[:, 2])
        return jnp.mean( (rv_pred*point_weights) ** 2)

    @property
    def loss_fns(self):
        return {
            "u_ic": self.loss_u_ic,
            "v_ic": self.loss_v_ic,
            "ru": self.loss_ru,
            "rv": self.loss_rv,
        }

class GinzburgLandauEvaluator(BaseEvaluator):
    def __init__(self, config, model, sol_dict):
        super().__init__(config, model)
        self.u_ref = sol_dict["u"]
        self.v_ref = sol_dict["v"]

    def log_errors(self, params):
        u_error, v_error = self.model.compute_l2_error(
            params,
            self.u_ref,
            self.v_ref,
        )
        self.log_dict["u_error"] = u_error
        self.log_dict["v_error"] = v_error
