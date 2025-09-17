import numpy as np
import scipy.stats as stats
from typing import Tuple

def logistic_weight(x, center, k=10, decreasing=False):
    val = 1 / (1 + np.exp(-k * (x - center)))
    return 1 - val if decreasing else val

def _get_stealthy_chameleon_params(x_val):
    x = float(x_val)

    M_global = 1.0 + np.sin(np.pi * x * 0.8)

    params_c1 = {}
    alpha_c1 = 20 * logistic_weight(x, center=0.25, k=15)
    scale_c1 = 0.1 + 3 * logistic_weight(x, center=0.15, k=10)
    if alpha_c1 == 0:
        delta_c1 = 0
    else:
        delta_c1 = alpha_c1 / np.sqrt(1 + alpha_c1**2)
    mean_offset_c1 = scale_c1 * delta_c1 * np.sqrt(2 / np.pi)
    loc_c1 = M_global - mean_offset_c1

    params_c1['type'] = 'skewnorm'
    params_c1['a'] = alpha_c1
    params_c1['loc'] = loc_c1
    params_c1['scale'] = scale_c1

    params_c2 = {}
    gap_c2 = 6 * logistic_weight(x, center=0.4, k=15)
    df_c2 = 1.2
    scale_val_c2 = 0.1 + 0.2 * logistic_weight(x, center=0.3, k=10)

    params_c2['type'] = 'bimodal_t'
    params_c2['loc_mean'] = M_global
    params_c2['gap'] = gap_c2
    params_c2['df'] = df_c2
    params_c2['scale'] = scale_val_c2

    params_c3 = {}
    a_beta_c3 = 0.2 + 0.3 * np.sin(np.pi * (x-0.6)*2.5 )
    b_beta_c3 = 0.2 + 0.3 * np.cos(np.pi * (x-0.6)*2.5 )
    a_beta_c3 = np.clip(a_beta_c3, 0.05, None)
    b_beta_c3 = np.clip(b_beta_c3, 0.05, None)

    M_c3 = M_global - 3 * logistic_weight(x, center=0.7, k=10)

    scale_beta_c3 = 2.0 + 2 * np.sin(np.pi * (x-0.6)*2)
    scale_beta_c3 = np.clip(scale_beta_c3, 0.5, None)

    if (a_beta_c3 + b_beta_c3) == 0:
        loc_beta_c3 = M_c3
    else:
        loc_beta_c3 = M_c3 - scale_beta_c3 * (a_beta_c3 / (a_beta_c3 + b_beta_c3))

    loc_t_c3 = M_c3 + 3 + 2*np.sin(np.pi*(x-0.6)*3)
    df_t_c3 = 1.5
    scale_t_c3 = 0.3 + 0.2*logistic_weight(x, 0.8, k=15)

    params_c3['type'] = 'beta_plus_t'
    params_c3['beta_a'] = a_beta_c3
    params_c3['beta_b'] = b_beta_c3
    params_c3['beta_loc'] = loc_beta_c3
    params_c3['beta_scale'] = scale_beta_c3
    params_c3['t_loc'] = loc_t_c3
    params_c3['t_df'] = df_t_c3
    params_c3['t_scale'] = scale_t_c3
    params_c3['beta_weight_internal'] = 0.85

    logit_w1 = 5 - 20 * np.abs(x - 0.2)
    logit_w2 = 5 - 20 * np.abs(x - 0.55)
    logit_w3 = 5 - 20 * np.abs(x - 0.85)

    logits = np.array([logit_w1, logit_w2, logit_w3])
    exp_logits = np.exp(logits - np.max(logits))
    weights = exp_logits / np.sum(exp_logits)

    return {
        'weights': weights,
        'c1': params_c1,
        'c2': params_c2,
        'c3': params_c3
    }

def generate_data(n_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate training data from the StealthyChameleon model."""
    rng = np.random.RandomState(seed)

    X_gen = rng.uniform(0, 1, n_samples).reshape(-1, 1)
    y_gen = np.zeros(n_samples)

    for i, x_val_arr in enumerate(X_gen):
        x_val = x_val_arr[0]
        p = _get_stealthy_chameleon_params(x_val)

        component_idx = rng.choice(3, p=p['weights'])

        if component_idx == 0:
            params = p['c1']
            y_gen[i] = stats.skewnorm.rvs(a=params['a'], loc=params['loc'], scale=params['scale'], random_state=rng)
        elif component_idx == 1:
            params = p['c2']
            if rng.rand() < 0.5:
                loc = params['loc_mean'] - params['gap']
            else:
                loc = params['loc_mean'] + params['gap']
            y_gen[i] = stats.t.rvs(df=params['df'], loc=loc, scale=params['scale'], random_state=rng)
        else:
            params = p['c3']
            if rng.rand() < params['beta_weight_internal']:
                try:
                    y_gen[i] = stats.beta.rvs(a=params['beta_a'], b=params['beta_b'],
                                              loc=params['beta_loc'], scale=params['beta_scale'], random_state=rng)
                except ValueError:
                     y_gen[i] = rng.uniform(params['beta_loc'], params['beta_loc'] + params['beta_scale'])
            else:
                y_gen[i] = stats.t.rvs(df=params['t_df'], loc=params['t_loc'], scale=params['t_scale'], random_state=rng)

    return X_gen, y_gen

def compute_true_quantiles(x_grid: np.ndarray, q_grid: np.ndarray, seed: int) -> np.ndarray:
    """Compute true conditional quantiles via simulation."""
    x_eval_grid = x_grid.flatten()
    q_eval_grid = np.atleast_1d(q_grid)

    rng = np.random.RandomState(seed)

    results = np.zeros((len(x_eval_grid), len(q_eval_grid)))
    n_sim_samples = 250000

    for i, x_val in enumerate(x_eval_grid):
        p = _get_stealthy_chameleon_params(x_val)
        sim_y_i = np.zeros(n_sim_samples)

        counts = rng.multinomial(n_sim_samples, p['weights'])

        current_idx = 0
        if counts[0] > 0:
            params_c1 = p['c1']
            sim_y_i[current_idx : current_idx+counts[0]] = stats.skewnorm.rvs(
                a=params_c1['a'], loc=params_c1['loc'], scale=params_c1['scale'], size=counts[0], random_state=rng)
            current_idx += counts[0]

        if counts[1] > 0:
            params_c2 = p['c2']
            n_mode1 = counts[1] // 2
            n_mode2 = counts[1] - n_mode1
            loc1 = params_c2['loc_mean'] - params_c2['gap']
            loc2 = params_c2['loc_mean'] + params_c2['gap']
            if n_mode1 > 0:
                sim_y_i[current_idx : current_idx+n_mode1] = stats.t.rvs(
                    df=params_c2['df'], loc=loc1, scale=params_c2['scale'], size=n_mode1, random_state=rng)
                current_idx += n_mode1
            if n_mode2 > 0:
                sim_y_i[current_idx : current_idx+n_mode2] = stats.t.rvs(
                    df=params_c2['df'], loc=loc2, scale=params_c2['scale'], size=n_mode2, random_state=rng)
                current_idx += n_mode2

        if counts[2] > 0:
            params_c3 = p['c3']
            n_beta_internal = int(counts[2] * params_c3['beta_weight_internal'])
            n_t_internal = counts[2] - n_beta_internal

            if n_beta_internal > 0:
                try:
                    sim_y_i[current_idx : current_idx+n_beta_internal] = stats.beta.rvs(
                        a=params_c3['beta_a'], b=params_c3['beta_b'],
                        loc=params_c3['beta_loc'], scale=params_c3['beta_scale'], size=n_beta_internal, random_state=rng)
                except ValueError:
                     sim_y_i[current_idx : current_idx+n_beta_internal] = rng.uniform(
                        params_c3['beta_loc'], params_c3['beta_loc'] + params_c3['beta_scale'], size=n_beta_internal)
                current_idx += n_beta_internal
            if n_t_internal > 0:
                sim_y_i[current_idx : current_idx+n_t_internal] = stats.t.rvs(
                    df=params_c3['t_df'], loc=params_c3['t_loc'], scale=params_c3['t_scale'], size=n_t_internal, random_state=rng)
                current_idx += n_t_internal

        results[i, :] = np.quantile(sim_y_i[:current_idx], q_eval_grid)

    return results
