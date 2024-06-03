# neural-MFG
code base for solving MFG using normalizing flow

`python -m venv ../venv/neural-MFG`

`source ../venv/neural-MFG/bin/activate`

`pip install -e .`

`pip install -r requirements.txt`

For switching between 1D and 2D test case, you need to modified the FLAGS.dim parameter and
*boundary condition on density* module and run:
`python src/mfg.py`

You can switch between three loss functions via modifying the **update** function:
```
# density fitting using conditional normalizing flow
loss, grads = jax.value_and_grad(density_fit_loss_fn)(params, rng, FLAGS.batch_size)

# solve wasserstein distance
# tuning the regularization constant weight inside the function
loss, grads = jax.value_and_grad(wasserstein_loss_fn)(params, rng, FLAGS.batch_size)

# solve mean-field game
# tunning the beta inside the function
loss, grads = jax.value_and_grad(mfg_loss_fn)(params, rng, FLAGS.batch_size)
```