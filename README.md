# cnf_ot
<!-- TODO: change the repo name, maybe cnf_mfc? -->
Code base for [Variational conditional normalizing flows for computing second-order mean field control problems](https://arxiv.org/abs/2503.19580)


## Quick start
1. Run the following script to prepare the environment
```bash
git clone git@github.com:jiaxi98/cnf_ot.git
python -m venv venv/cnf_ot
source venv/cnf_ot/bin/activate
cd cnf_ot
pip install .
pip install -r requirements.txt
```
2. Modify config/mfc.yaml for different settings (see following instructions
) and solve the problem via:
```python
python cnf_ot/solvers.py
```

## Specify problem parameters
Our codebase supports solving three kinds of problems: **optimal transport,
regularized Wasserstein proximal operator, and Fokker-Planck equation**. Check
our paper for mathematical formulations of these problems and the parameters.

*optimal transport*: switch type to ``ot'' to solve
```ymal
type: ot # ot, rwpo, fp
ot:
  subtype: free # free, obstacle
```
subtype "free" corresponds to the original optimal transport problem while subtype
"obstacle" corresponds to optimal transport with a soft obstacle between
source and target, passing through which has an additional cost.

*regularized Wasserstein proximal operator*: switch type to ``rwpo'' to solve
```ymal
type: rwpo # ot, rwpo, fp
rwpo:
  T: 1
  beta: 1
  a: 1
  pot_type: quadratic # quadratic, double_well
```
$T$ is the length of the trajectory, $\beta$ is the diffusion strength. Two
types of potential functions are supported, e.g. quadratic potential and
double-well potential with $-a, a$ the positions of two wells.

*Fokker-Planck equation*: switch type to ``fp'' to solve
```ymal
type: fp # ot, rwpo, fp
fp:
  T: 1
  a: 1 # drift coeff
  sigma: .5
  velocity_field_type: gradient # gradient, nongradient
```
$T$ is the simulation time, $\sigma$ is the diffusion strength, and $a$ is
the drift coefficient. Two types of drift vector fields are supported, e.g.
gradient velocity field with drift coefficient $a$ and non-gradient vector
field.