# Third-party components

- [verl](https://github.com/verl-project/verl), revision
  `b9d71f9a84ef89ec7f5a946cd277b35165a3daae`, is an external training
  dependency licensed under Apache-2.0. `patches/verl-compat.patch` contains
  the source checkout's compatibility changes for chat templates, text-only
  SFT datasets. Preserve upstream notices when using it.
- `third_party/repo_predictors/multiprop_utils/` contains predictor adapters
  derived from [RePO](https://github.com/tmlr-group/RePO), with local API and
  scikit-learn compatibility changes. The source RePO checkout identified
  revision `c2f026fadadb17fb749e7d876fc8b5d4fd66b64f` and contained local changes.
  These files retain their original implementation; inclusion here does not
  relicense upstream code or predictor assets.
- `proper_utils.py` includes synthetic-accessibility scoring derived from
  RDKit's SA scorer. RDKit and its contributed scoring code retain their
  respective notices and terms.
- MuMOInstruct data, ADMET-AI weights, the DRD2 classifier, and fingerprint
  scoring data are external assets. Obtain them from their original providers
  and follow their respective terms. No dataset or model weights are bundled.

A license for the original MARCO contribution has not yet been selected.
No additional reuse license is implied by this private publication snapshot.
