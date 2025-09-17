import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax import random, jit, grad, value_and_grad
from flax import linen as nn
from flax.training import train_state

class IQNRegressor:
    def __init__(self, config=None, **kwargs):

        default_config = {
            "learning_rate": 1e-3,
            "num_epochs": 20,
            "batch_size": 64,
            "quantile_samples_per_datum": 1
        }


        if config is None:
            config = {}

        for key, value in default_config.items():
            config.setdefault(key, value)

        self.config = config


    def fit(self, X, y):

        y = y.reshape(-1, 1)
        X_ref = jnp.concatenate([X,y], axis=1)

        self.state = train_model(X_ref, self.config['num_epochs'],
                                 self.config['batch_size'],
                                 self.config['learning_rate'],
                                 random.PRNGKey(0),
                                 samples_per_datum=self.config['quantile_samples_per_datum'])

    def predict(self, X, q, **kwargs):
        n, d = X.shape
        k = len(q)
        X_repeat = jnp.repeat(X, k, axis=0) #Shape (n*k, d)
        q_tiled = jnp.tile(q, n).reshape(-1,1) #Shape (n*k,1)
        X_ref = jnp.concatenate([X_repeat, q_tiled], axis=1) #Shape (n*k, d+1)

        preds = nn_predict(self.state, X_ref)
        preds_reshaped = preds.reshape(n, k)

        return preds_reshaped



class IQN(nn.Module):
    input_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x, tau):
        combined_input = jnp.concatenate([x, tau], axis=-1)
        x = nn.Dense(features=128)(combined_input)
        x = nn.leaky_relu(x)
        x = nn.Dense(features=256)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(features=512)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(features=512)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(features=self.output_dim)(x)
        return x

def implicit_quantile_loss(taus, targets, predictions):
    taus = jnp.ravel(taus)
    targets = jnp.ravel(targets)
    predictions = jnp.ravel(predictions)

    errors = targets - predictions
    losses = jnp.maximum((taus - 1) * errors, taus * errors)
    return jnp.sum(losses)

def create_train_state(rng, learning_rate, input_dim, output_dim):
    model = IQN(input_dim=input_dim, output_dim=output_dim)
    params = model.init(rng, jnp.ones([1, input_dim]), jnp.ones([1, 1]))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jit
def train_step(state, batch):
    def loss_fn(params):
        x, y, tau = batch
        y_pred = state.apply_fn(params, x, tau)
        loss = implicit_quantile_loss(tau, y, y_pred)
        return loss

    loss, grads = value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def data_loader(X, y, tau, batch_size, shuffle=True):
    n = X.shape[0]
    indices = jnp.arange(n)
    if shuffle:
        indices = jax.random.permutation(random.PRNGKey(0), indices)
    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices], tau[batch_indices]

def train_model(X_ref, num_epochs, batch_size, learning_rate, rng, samples_per_datum=1):

    # Repeat X_ref samples_per_datum times along axis 0
    X_ref = jnp.repeat(X_ref, samples_per_datum, axis=0)
    n_ref = X_ref.shape[0]
    p = X_ref.shape[1] - 1
    y_arr = jnp.array(X_ref[:, -1], dtype=jnp.float32).reshape(-1, 1)
    x_arr = jnp.array(X_ref[:, :-1], dtype=jnp.float32)
    tau_arr = jax.random.uniform(random.PRNGKey(0), (n_ref, 1))


    state = create_train_state(rng, learning_rate, p, 1)

    for epoch in range(num_epochs):
        total_loss = 0
        for x_batch, y_batch, tau_batch in data_loader(x_arr, y_arr, tau_arr, batch_size):
            x_batch, y_batch, tau_batch = jnp.array(x_batch), jnp.array(y_batch), jnp.array(tau_batch)
            state, loss = train_step(state, (x_batch, y_batch, tau_batch))
            total_loss += loss

        avg_loss = total_loss / (n_ref // batch_size)
        # print(f'Epoch {epoch+1}, Loss: {avg_loss}')

    return state

def nn_predict(state, X_new):
    """
    Predicts the output for new data using the trained model state.

    Parameters:
    - state: The trained model state.
    - X_new: New input data of shape (n, p + 1), where the last column is tau.

    Returns:
    - predictions: The predicted outputs.
    """
    x_new = jnp.array(X_new[:, :-1], dtype=jnp.float32)
    tau_new = jnp.array(X_new[:, -1], dtype=jnp.float32).reshape(-1, 1)
    predictions = state.apply_fn(state.params, x_new, tau_new)
    return predictions

def sample_predictive(state, X_covariates, n_samples, rng):
    """
    Generates samples from the predictive posterior distribution.

    Parameters:
    - state: The trained model state.
    - X_covariates: Covariates of shape (n, p).
    - n_samples: Number of samples to generate.
    - rng: JAX random number generator key.

    Returns:
    - predictions: Array of shape (n, n_samples) containing the predictive samples.
    """
    n, p = X_covariates.shape

    # Generate n_samples tau values for each row
    taus = jax.random.uniform(rng, (n, n_samples))

    # Repeat the covariates for each tau sample
    X_repeated = jnp.repeat(X_covariates, n_samples, axis=0)
    taus_flat = taus.flatten()[:, None]  # Flatten taus and make it a column vector

    # Concatenate the repeated covariates with the flattened taus
    X_combined = jnp.concatenate([X_repeated, taus_flat], axis=1)

    # Predict using the combined covariates and taus
    predictions_flat = predict(state, X_combined)

    # Reshape the predictions to (n, n_samples)
    predictions = predictions_flat.reshape(n, n_samples)

    return predictions


def fit_predict(X_train: np.ndarray, y_train: np.ndarray,
                x_grid: np.ndarray, q_grid: np.ndarray,
                hidden_sizes: list = [100, 100],
                learning_rate: float = 0.001,
                batch_size: int = 32,
                epochs: int = 200,
                quantile_samples_per_datum: int = 1,
                **kwargs) -> np.ndarray:
    """Fit IQN model and predict on evaluation grid."""

    config = {
        "learning_rate": learning_rate,
        "num_epochs": epochs,
        "batch_size": batch_size,
        "quantile_samples_per_datum": quantile_samples_per_datum
    }

    model = IQNRegressor(config=config)
    model.fit(X_train, y_train)

    predictions = model.predict(x_grid, q_grid)

    return predictions