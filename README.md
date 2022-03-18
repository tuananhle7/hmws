# Hybrid Memoised Wake Sleep

Code for [Hybrid Memoised Wake-Sleep](https://openreview.net/forum?id=auOPcdAcoy).
Cleaning up still in progress.

Bibtex:
```
@inproceedings{le2022hybrid,
  title={Hybrid Memoised Wake-Sleep: Approximate Inference at the Discrete-Continuous Interface},
  author={Tuan Anh Le and Katherine M. Collins and Luke Hewitt and Kevin Ellis and Siddharth N and Samuel Gershman and Joshua B. Tenenbaum},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=auOPcdAcoy}
}
```

## Installation

```
git clone git@github.com:tuananhle7/cmws.git
cd cmws
pip install .
```
or
```
pip install -e .
```
for editable mode.

Install Sid's "rws" branch of pyro
```
pip install git+https://github.com/iffsid/pyro.git@rws
```

## Unit tests
```
pytest tests
```
