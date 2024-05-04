# neural-MFG
code base for solving MFG using normalizing flow

`python -m venv ../venv/neural-MFG`

`source ../venv/neural-MFG/bin/activate`

`pip install -e .`

`pip install -r requirements.txt`

For switching between 1D and 2D test case, you need to modified the FLAGS.dim parameter and
*boundary condition on density* module and run:
`python tests/test_wasserstein_geodesic.py`