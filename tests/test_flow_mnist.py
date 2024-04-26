"""A simple example of a flow model trained on MNIST."""

from typing import Iterator, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorflow_datasets as tfds
from absl import app, flags, logging
from jaxtyping import Array

from src.flows import RQSFlow
from src.types import Batch, OptState, PRNGKey

flags.DEFINE_integer(
  "flow_num_layers", 8, "Number of layers to use in the flow."
)
flags.DEFINE_integer(
  "mlp_num_layers", 2, "Number of layers to use in the MLP conditioner."
)
flags.DEFINE_integer("hidden_size", 500, "Hidden size of the MLP conditioner.")
flags.DEFINE_integer(
  "num_bins", 4, "Number of bins to use in the rational-quadratic spline."
)
flags.DEFINE_integer(
  "batch_size", 128, "Batch size for training and evaluation."
)
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 5000, "Number of training steps to run.")
flags.DEFINE_integer("eval_frequency", 100, "How often to evaluate the model.")
FLAGS = flags.FLAGS

MNIST_IMAGE_SHAPE = (28, 28, 1)


def load_dataset(split: tfds.Split, batch_size: int) -> Iterator[Batch]:
  ds = tfds.load("mnist", split=split, shuffle_files=True)
  ds = ds.shuffle(buffer_size=10 * batch_size)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=5)
  ds = ds.repeat()
  return iter(tfds.as_numpy(ds))


def prepare_data(batch: Batch, prng_key: Optional[PRNGKey] = None) -> Array:
  data = batch["image"].astype(np.float32)
  if prng_key is not None:
    # Dequantize pixel values {0, 1, ..., 255} with uniform noise [0, 1).
    data += jax.random.uniform(prng_key, data.shape)
  return data / 256.  # Normalize pixel values from [0, 256) to [0, 1).


def show_samples(samples: Array, rows=2):
  n_samples = len(samples)
  cols = n_samples // rows
  fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
  for i in range(rows):
    for j in range(cols):
      ax = axes[i, j]
      idx = i * cols + j
      if idx < n_samples:
        ax.imshow(samples[idx], cmap='gray', interpolation='none')
        ax.set_title(f'Image {idx}')
      ax.axis('off')  # Turn off axis numbers and ticks
  plt.tight_layout()
  plt.show()


def main(_):
  optimizer = optax.adam(FLAGS.learning_rate)
  prng_seq = hk.PRNGSequence(42)

  model = RQSFlow(
    event_shape=MNIST_IMAGE_SHAPE,
    num_layers=FLAGS.flow_num_layers,
    hidden_sizes=[FLAGS.hidden_size] * FLAGS.mlp_num_layers,
    num_bins=FLAGS.num_bins
  )

  params = model.init(next(prng_seq), np.zeros((1, *MNIST_IMAGE_SHAPE)))
  opt_state = optimizer.init(params)

  def loss_fn(params: hk.Params, prng_key: PRNGKey, batch: Batch) -> Array:
    data = prepare_data(batch, prng_key)
    # Loss is average negative log likelihood.
    loss = -jnp.mean(model.apply.log_prob(params, data))
    return loss

  @jax.jit
  def update(
    params: hk.Params, prng_key: PRNGKey, opt_state: OptState, batch: Batch
  ) -> Tuple[hk.Params, OptState]:
    """Single SGD update step."""
    grads = jax.grad(loss_fn)(params, prng_key, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

  @jax.jit
  def eval_fn(params: hk.Params, batch: Batch) -> Array:
    data = prepare_data(batch)  # We don't dequantize during evaluation.
    loss = -jnp.mean(model.apply.log_prob(params, data))
    return loss

  train_ds = load_dataset(tfds.Split.TRAIN, FLAGS.batch_size)
  valid_ds = load_dataset(tfds.Split.TEST, FLAGS.batch_size)

  sample_fn = jax.jit(model.apply.sample, static_argnames=['sample_shape'])

  n_samples = 10
  samples = sample_fn(params, seed=next(prng_seq), sample_shape=(n_samples, ))
  samples = samples * 256  # Denormalize pixel values from [0, 1) to [0, 256)
  show_samples(samples)

  batch = next(train_ds)
  show_samples(batch['image'][:10])

  for step in range(FLAGS.training_steps):
    batch = next(train_ds)
    prng_key = next(prng_seq)

    params, opt_state = update(params, prng_key, opt_state, batch)

    if step % FLAGS.eval_frequency == 0:
      val_loss = eval_fn(params, next(valid_ds))
      logging.info("STEP: %5d; Validation loss: %.3f", step, val_loss)

  batch = next(valid_ds)
  show_samples(batch['image'][:10])

  n_samples = 10
  samples = sample_fn(params, seed=next(prng_seq), sample_shape=(n_samples, ))
  samples = samples * 256  # Denormalize pixel values from [0, 1) to [0, 256)
  show_samples(samples)


if __name__ == "__main__":
  app.run(main)
