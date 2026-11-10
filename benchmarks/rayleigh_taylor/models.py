from functools import partial
import jax.numpy as jnp
from jax import jit, vmap, jacfwd
from pipeline.models import PINN
from pipeline.evaluator import BaseEvaluator


class NavierStokes(PINN):
    # residual families in the order r_net returns them (loss-weight keys)
    residual_families = ("ru", "rv", "rc", "re")

    def __init__(self, config, axes_dict, initial_dict, pde_params_dict):
        super().__init__(
            config=config,
            axes_dict=axes_dict,
            initial_dict=initial_dict,
            pde_params_dict=pde_params_dict,
        )

        self.alpha1 = pde_params_dict["alpha1"]
        self.alpha2 = pde_params_dict["alpha2"]
        self.alpha3 = pde_params_dict["alpha3"]
        self.alpha4 = pde_params_dict["alpha4"]

        self.u0 = initial_dict["u0"]
        self.v0 = initial_dict["v0"]
        self.p0 = initial_dict["p0"]
        self.temp0 = initial_dict["T0"]

        self.t_star = axes_dict["t"]
        self.coords_x = axes_dict["coords"][:, 0]
        self.coords_y = axes_dict["coords"][:, 1]

        velocity_scale = jnp.max(jnp.sqrt(self.u0 ** 2 + self.v0 ** 2))
        self.velocity_scale = velocity_scale + 1e-3

        # Residuals returned as shape (4,)
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0, 0))

        self.ic_pred_fn = vmap(self.nn_pred_vec, (None, None, 0, 0))              # (N, 4)
        self.bc_pred_fn = vmap(self.nn_pred_vec, (None, 0, 0, 0))                 # (N, 4)
        self.domain_pred_fn = vmap(vmap(self.nn_pred_vec, (None, None, 0, 0)),
                                (None, 0, None, None))                         # (T, N, 4)

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

    def p_net(self, params, t, x, y):
        return self.nn_pred_vec(params, t, x, y)[2]

    def temp_net(self, params, t, x, y):
        return self.nn_pred_vec(params, t, x, y)[3]

    def r_net(self, params, t, x, y):
        """Efficient compute of the residual loss."""
        f = lambda t_, x_, y_: self.nn_pred_vec(params, t_, x_, y_)

        y0 = f(t, x, y)  # shape (4,)
        u, v, p, temp = y0

        # First derivatives wrt (t, x, y)
        # jacfwd over argnums=(0,1,2) returns a tuple of 3 arrays, each shape (4,)
        J_cols = jacfwd(f, argnums=(0, 1, 2))(t, x, y)
        J = jnp.stack(J_cols, axis=1)  # shape (4, 3), columns = [t, x, y]

        # Spatial Hessian wrt (x, y) only
        g = lambda x_, y_: f(t, x_, y_)
        H_cols = jacfwd(jacfwd(g, argnums=(0, 1)), argnums=(0, 1))(x, y)
        # H_cols is a tuple over outer deriv index, each entry a tuple over inner deriv index
        # each leaf has shape (4,)
        H = jnp.stack(
            [jnp.stack(H_row, axis=1) for H_row in H_cols],
            axis=1,
        )  # shape (4, 2, 2)

        u_t, u_x, u_y = J[0, 0], J[0, 1], J[0, 2]
        v_t, v_x, v_y = J[1, 0], J[1, 1], J[1, 2]
        p_x, p_y = J[2, 1], J[2, 2]
        temp_t, temp_x, temp_y = J[3, 0], J[3, 1], J[3, 2]

        u_lap = H[0, 0, 0] + H[0, 1, 1]
        v_lap = H[1, 0, 0] + H[1, 1, 1]
        temp_lap = H[3, 0, 0] + H[3, 1, 1]

        ru = u_t + u * u_x + v * u_y + p_x - self.alpha1 * u_lap
        rv = v_t + u * v_x + v * v_y + p_y - self.alpha1 * v_lap - self.alpha2 * temp
        rc = u_x + v_y
        re = temp_t + u * temp_x + v * temp_y - self.alpha4 * temp_lap

        return jnp.array([ru, rv, rc, re])

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, t_star, u_ref, v_ref, temp_ref):
        pred = self.domain_pred_fn(params, t_star, self.coords_x, self.coords_y)

        u_pred = pred[..., 0]
        v_pred = pred[..., 1]
        temp_pred = pred[..., 3]

        u_error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
        v_error = jnp.linalg.norm(v_pred - v_ref) / jnp.linalg.norm(v_ref)
        temp_error = jnp.linalg.norm(temp_pred - temp_ref) / jnp.linalg.norm(temp_ref)

        return u_error, v_error, temp_error

    @partial(jit, static_argnums=(0,))
    def r_losses_pp(self, params, candidate_batch, weights):
        """Compute the PDE residual loss for a batch of points weighted by the losses of the governing terms"""
        t = candidate_batch[:, 0]
        x = candidate_batch[:, 1]
        y = candidate_batch[:, 2]
        r_pred = self.r_pred_fn(params, t, x, y)   # (N, 4)
        return (
            r_pred[:, 0] ** 2 * weights["ru"]
            + r_pred[:, 1] ** 2 * weights["rv"]
            + r_pred[:, 2] ** 2 * weights["rc"]
            + r_pred[:, 3] ** 2 * weights["re"]
        )

    @partial(jit, static_argnums=(0,))
    def loss(self, params, weights, batch, point_weights):
        t_bc = batch[:, 0]
        x_bc = batch[:, 1]

        ic_pred = self.ic_pred_fn(params, 0.0, self.coords_x, self.coords_y)
        u_ic_loss = jnp.mean((ic_pred[:, 0] - self.u0) ** 2)
        v_ic_loss = jnp.mean((ic_pred[:, 1] - self.v0) ** 2)
        temp_ic_loss = jnp.mean((ic_pred[:, 3] - self.temp0) ** 2)

        y0 = jnp.zeros_like(x_bc)
        y1 = 2.0 * jnp.ones_like(x_bc)

        bc0_pred = self.bc_pred_fn(params, t_bc, x_bc, y0)
        bc1_pred = self.bc_pred_fn(params, t_bc, x_bc, y1)

        u_bc_loss = 0.5 * (jnp.mean(bc0_pred[:, 0] ** 2) + jnp.mean(bc1_pred[:, 0] ** 2))
        v_bc_loss = 0.5 * (jnp.mean(bc0_pred[:, 1] ** 2) + jnp.mean(bc1_pred[:, 1] ** 2))
        temp_bc_loss = 0.5 * (jnp.mean(bc0_pred[:, 3] ** 2) + jnp.mean(bc1_pred[:, 3] ** 2))

        # Residual losses
        r_pred = self.r_pred_fn(params, batch[:, 0], batch[:, 1], batch[:, 2])  # shape (N, 4)
        r_losses = jnp.mean( (point_weights[:, None] * r_pred) ** 2, axis=0)

        return (
            weights["u_ic"] * u_ic_loss
            + weights["v_ic"] * v_ic_loss
            + weights["temp_ic"] * temp_ic_loss
            + weights["u_bc"] * u_bc_loss
            + weights["v_bc"] * v_bc_loss
            + weights["temp_bc"] * temp_bc_loss
            + weights["ru"] * r_losses[0]
            + weights["rv"] * r_losses[1]
            + weights["rc"] * r_losses[2]
            + weights["re"] * r_losses[3]
        )
    
    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch, point_weights):
        t_bc = batch[:, 0]
        x_bc = batch[:, 1]

        ic_pred = self.ic_pred_fn(params, 0.0, self.coords_x, self.coords_y)
        u_ic_loss = jnp.mean((ic_pred[:, 0] - self.u0) ** 2)
        v_ic_loss = jnp.mean((ic_pred[:, 1] - self.v0) ** 2)
        temp_ic_loss = jnp.mean((ic_pred[:, 3] - self.temp0) ** 2)

        y0 = jnp.zeros_like(x_bc)
        y1 = 2.0 * jnp.ones_like(x_bc)

        bc0_pred = self.bc_pred_fn(params, t_bc, x_bc, y0)
        bc1_pred = self.bc_pred_fn(params, t_bc, x_bc, y1)

        u_bc_loss = 0.5 * (jnp.mean(bc0_pred[:, 0] ** 2) + jnp.mean(bc1_pred[:, 0] ** 2))
        v_bc_loss = 0.5 * (jnp.mean(bc0_pred[:, 1] ** 2) + jnp.mean(bc1_pred[:, 1] ** 2))
        temp_bc_loss = 0.5 * (jnp.mean(bc0_pred[:, 3] ** 2) + jnp.mean(bc1_pred[:, 3] ** 2))

        # Residual losses
        r_pred = self.r_pred_fn(params, batch[:, 0], batch[:, 1], batch[:, 2])  # shape (N, 4)
        r_losses = jnp.mean((point_weights[:, None] * r_pred) ** 2, axis=0)

        return {
            "u_ic": u_ic_loss,
            "v_ic": v_ic_loss,
            "temp_ic": temp_ic_loss,
            "u_bc": u_bc_loss,
            "v_bc": v_bc_loss,
            "temp_bc": temp_bc_loss,
            "ru": r_losses[0],
            "rv": r_losses[1],
            "rc": r_losses[2],
            "re": r_losses[3]
        }


    @partial(jit, static_argnums=(0,))
    def pred_last_t(self, params):
        pred = self.ic_pred_fn(params, self.t_star[-1], self.coords_x, self.coords_y)
        return {
            "u0": pred[:, 0],
            "v0": pred[:, 1],
            "p0": pred[:, 2],
            "T0": pred[:, 3],
        }

    @partial(jit, static_argnums=(0,))
    def pred_domain(self, params):
        pred = self.domain_pred_fn(params, self.t_star, self.coords_x, self.coords_y)
        return {
            "u": pred[..., 0],
            "v": pred[..., 1],
            "p": pred[..., 2],
            "T": pred[..., 3],
        }

    """Splitted loss functions for more efficient compute of the individual grads of the loss_fns"""
    @partial(jit, static_argnums=(0,))
    def gov_u_loss_point(self, params, t, x, y):
        u = self.u_net(params, t, x, y)
        v = self.v_net(params, t, x, y)

        u_t = jacfwd(self.u_net, argnums=1)(params, t, x, y)
        u_x = jacfwd(self.u_net, argnums=2)(params, t, x, y)
        u_y = jacfwd(self.u_net, argnums=3)(params, t, x, y)

        p_x = jacfwd(self.p_net, argnums=2)(params, t, x, y)

        u_xx = jacfwd(jacfwd(self.u_net, argnums=2), argnums=2)(params, t, x, y)
        u_yy = jacfwd(jacfwd(self.u_net, argnums=3), argnums=3)(params, t, x, y)

        ru = u_t + u * u_x + v * u_y + p_x - self.alpha1 * (u_xx + u_yy)
        return ru

    @partial(jit, static_argnums=(0,))
    def gov_v_loss_point(self, params, t, x, y):
        u = self.u_net(params, t, x, y)
        v = self.v_net(params, t, x, y)
        temp = self.temp_net(params, t, x, y)

        v_t = jacfwd(self.v_net, argnums=1)(params, t, x, y)
        v_x = jacfwd(self.v_net, argnums=2)(params, t, x, y)
        v_y = jacfwd(self.v_net, argnums=3)(params, t, x, y)

        p_y = jacfwd(self.p_net, argnums=3)(params, t, x, y)

        v_xx = jacfwd(jacfwd(self.v_net, argnums=2), argnums=2)(params, t, x, y)
        v_yy = jacfwd(jacfwd(self.v_net, argnums=3), argnums=3)(params, t, x, y)

        rv = v_t + u * v_x + v * v_y + p_y - self.alpha1 * (v_xx + v_yy) - self.alpha2 * temp
        return rv

    @partial(jit, static_argnums=(0,))
    def gov_c_loss_point(self, params, t, x, y):
        u_x = jacfwd(self.u_net, argnums=2)(params, t, x, y)
        v_y = jacfwd(self.v_net, argnums=3)(params, t, x, y)
        rc = u_x + v_y
        return rc

    @partial(jit, static_argnums=(0,))
    def gov_temp_loss_point(self, params, t, x, y):
        u = self.u_net(params, t, x, y)
        v = self.v_net(params, t, x, y)

        temp_t = jacfwd(self.temp_net, argnums=1)(params, t, x, y)
        temp_x = jacfwd(self.temp_net, argnums=2)(params, t, x, y)
        temp_y = jacfwd(self.temp_net, argnums=3)(params, t, x, y)

        temp_xx = jacfwd(jacfwd(self.temp_net, argnums=2), argnums=2)(params, t, x, y)
        temp_yy = jacfwd(jacfwd(self.temp_net, argnums=3), argnums=3)(params, t, x, y)

        re = temp_t + u * temp_x + v * temp_y - self.alpha4 * (temp_xx + temp_yy)
        return re


    @partial(jit, static_argnums=(0,))
    def loss_gov_u(self, params, batch, point_weights):
        ru = vmap(self.gov_u_loss_point, (None, 0, 0, 0))(
            params, batch[:, 0], batch[:, 1], batch[:, 2]
        )
        return jnp.mean((ru*point_weights) ** 2)

    @partial(jit, static_argnums=(0,))
    def loss_gov_v(self, params, batch, point_weights):
        rv = vmap(self.gov_v_loss_point, (None, 0, 0, 0))(
            params, batch[:, 0], batch[:, 1], batch[:, 2]
        )
        return jnp.mean((rv*point_weights) ** 2)

    @partial(jit, static_argnums=(0,))
    def loss_gov_c(self, params, batch, point_weights):
        rc = vmap(self.gov_c_loss_point, (None, 0, 0, 0))(
            params, batch[:, 0], batch[:, 1], batch[:, 2]
        )
        return jnp.mean((rc*point_weights) ** 2)

    @partial(jit, static_argnums=(0,))
    def loss_gov_temp(self, params, batch, point_weights):
        re = vmap(self.gov_temp_loss_point, (None, 0, 0, 0))(
            params, batch[:, 0], batch[:, 1], batch[:, 2]
        )
        return jnp.mean((re*point_weights) ** 2)

    @partial(jit, static_argnums=(0,))
    def loss_u_ic(self, params, batch, point_weights):
        pred = self.ic_pred_fn(params, 0.0, self.coords_x, self.coords_y)
        return jnp.mean((pred[:, 0] - self.u0) ** 2)

    @partial(jit, static_argnums=(0,))
    def loss_v_ic(self, params, batch, point_weights):
        pred = self.ic_pred_fn(params, 0.0, self.coords_x, self.coords_y)
        return jnp.mean((pred[:, 1] - self.v0) ** 2)

    @partial(jit, static_argnums=(0,))
    def loss_temp_ic(self, params, batch, point_weights):
        pred = self.ic_pred_fn(params, 0.0, self.coords_x, self.coords_y)
        return jnp.mean((pred[:, 3] - self.temp0) ** 2)

    @partial(jit, static_argnums=(0,))
    def loss_u_bc(self, params, batch, point_weights):
        t_bc = batch[:, 0]
        x_bc = batch[:, 1]
        y0 = jnp.zeros_like(x_bc)
        y1 = 2.0 * jnp.ones_like(x_bc)

        bc0 = self.bc_pred_fn(params, t_bc, x_bc, y0)
        bc1 = self.bc_pred_fn(params, t_bc, x_bc, y1)

        return 0.5 * (jnp.mean(bc0[:, 0] ** 2) + jnp.mean(bc1[:, 0] ** 2))

    @partial(jit, static_argnums=(0,))
    def loss_v_bc(self, params, batch, point_weights):
        t_bc = batch[:, 0]
        x_bc = batch[:, 1]
        y0 = jnp.zeros_like(x_bc)
        y1 = 2.0 * jnp.ones_like(x_bc)

        bc0 = self.bc_pred_fn(params, t_bc, x_bc, y0)
        bc1 = self.bc_pred_fn(params, t_bc, x_bc, y1)

        return 0.5 * (jnp.mean(bc0[:, 1] ** 2) + jnp.mean(bc1[:, 1] ** 2))

    @partial(jit, static_argnums=(0,))
    def loss_temp_bc(self, params, batch, point_weights):
        t_bc = batch[:, 0]
        x_bc = batch[:, 1]
        y0 = jnp.zeros_like(x_bc)
        y1 = 2.0 * jnp.ones_like(x_bc)

        bc0 = self.bc_pred_fn(params, t_bc, x_bc, y0)
        bc1 = self.bc_pred_fn(params, t_bc, x_bc, y1)

        return 0.5 * (jnp.mean(bc0[:, 3] ** 2) + jnp.mean(bc1[:, 3] ** 2))

    @property
    def loss_fns(self):
        return {
            "u_ic": self.loss_u_ic,
            "v_ic": self.loss_v_ic,
            "temp_ic": self.loss_temp_ic,
            "u_bc": self.loss_u_bc,
            "v_bc": self.loss_v_bc,
            "temp_bc": self.loss_temp_bc,
            "ru": self.loss_gov_u,
            "rv": self.loss_gov_v,
            "rc": self.loss_gov_c,
            "re": self.loss_gov_temp,
        }
    
class NavierStokesEvaluator(BaseEvaluator):
    """Relative L2 errors of u, v and T over the whole (t, mesh) reference.
    `l2_error` (logged first, so it is the scalar the frozen-batch study tracks)
    is the mean of the three; the per-field errors follow."""

    def __init__(self, config, model, sol_dict):
        super().__init__(config, model)
        self.u_ref = sol_dict["u"]
        self.v_ref = sol_dict["v"]
        self.temp_ref = sol_dict["T"]
        # the evaluation times: the model's (single-window training path does not
        # put "t" into sol_dict; the time-window path does)
        self.t_star = sol_dict["t"] if "t" in sol_dict else model.t_star

    def log_errors(self, params):
        u_error, v_error, temp_error = self.model.compute_l2_error(
            params,
            self.t_star,
            self.u_ref,
            self.v_ref,
            self.temp_ref,
        )
        self.log_dict["l2_error"] = (u_error + v_error + temp_error) / 3.0
        self.log_dict["u_error"] = u_error
        self.log_dict["v_error"] = v_error
        self.log_dict["temp_error"] = temp_error
