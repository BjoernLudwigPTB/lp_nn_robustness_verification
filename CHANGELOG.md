# Changelog

<!--next-version-placeholder-->

## v0.8.0 (2023-01-24)
### Feature
* **CITATION:** Introduce reference to PySCIPOpt and data ([`4c1d4bb`](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/commit/4c1d4bb28f3b4f9b6d38295bdd0bdcca24e9041b))
* **timing_evaluation notebook:** Introduce timing evaluation in jupyter notebook ([`a84e2cf`](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/commit/a84e2cff5ea448a6f03c69bf9d390e3e3a8f1c60))

### Fix
* **pre_processing:** Replace z_i by theta_i in the calculation of xi_i ([`cc1d9c8`](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/commit/cc1d9c8941ec49bb72de2faf03086838bedc3377))
* Adapt all calls of ZeMASamples to most recent version v0.7.0 of zema_emc_annotated ([`6807f6b`](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/commit/6807f6b919a18b38f6b3c826faff8116ca5b3ba4))

### Documentation
* **timing_evaluation notebook:** Introduce timing evaluation jupyter notebook ([`56e7bcb`](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/commit/56e7bcb59bc439337d302fdadd601777f51add75))

**[See all commits in this version](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/compare/v0.7.0...v0.8.0)**

## v0.7.0 (2023-01-20)
### Feature
* **pre_processing:** Utilize timing module in pre-processing ([`c7926cb`](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/commit/c7926cbed7bb378d01006955b352f76406a32360))
* **timing:** Introduce module to time and store progress ([`dddd24d`](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/commit/dddd24d129ddb533ca1d286531384c4fc60f80ca))

### Documentation
* **timing:** Introduce timing module into docs ([`5670c98`](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/commit/5670c982382ecc0a627ef5307c35f55e68235133))

**[See all commits in this version](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/compare/v0.6.1...v0.7.0)**

## v0.6.1 (2023-01-19)
### Fix
* **linear_program:** Remove any clutter from linear_program after finding same results in all cases ([`e4d1f2f`](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/commit/e4d1f2fa6d51e7b07402609693c2f125941d7e11))
* **INSTALL.md:** Fix a bunch of formatting and information issues ([`8a1522b`](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/commit/8a1522b07d61b8b1adb57cdaf656533ce4218d95))

### Documentation
* **examples:** Introduce all examples code into docs ([`5d4ddd3`](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/commit/5d4ddd3c3563438a7938de339a17ef181d11be5a))

**[See all commits in this version](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/compare/v0.6.0...v0.6.1)**

## v0.6.0 (2023-01-19)
### Feature
* **CITATION.cff:** Introduce DOI ([`942bea6`](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/commit/942bea652e40decc9e915d2deb50d423c6d7ad9e))

### Fix
* **README:** Finalize README with mandatory sections ([`0c6d01b`](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/commit/0c6d01b08bf1d51a6f6cc6fbc68e667e25b73d56))

### Documentation
* **INSTALL:** Finalize installation instructions ([`cf66be8`](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/commit/cf66be80a30f8bed567b706b39f97a9115ca90a1))
* **README:** Introduce DOI and GitHub release badge ([`52f2477`](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/commit/52f24773084b2d8c54f69c2571af5035c5513bd6))

**[See all commits in this version](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/compare/v0.5.0...v0.6.0)**

## v0.5.0 (2023-01-19)
### Feature
* **activation_functions:** Introduce Identity activation function ([`9dac464`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/9dac46416dd535e6b3bafa3ae596e880fb04bf77))
* **linear_program:** Replace auxiliary variable by direct objective and thus fix residing issue ([`27021f4`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/27021f47b6e8c2ca184c1694af4572dd932cbbb9))
* **linear_program:** Introduce generic base class and two classes for original and adapted problem ([`a17389e`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/a17389e20d77305401ead6d66356d976ac7f9980))
* **data_types:** Introduce IndexAndSeed data_type for streamlined implementation of example ([`03c5123`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/03c512336d684605845655168c3a186e206cbad7))
* **examples:** Introduce task id parameter to parallelize and streamline implementation ([`e82582b`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/e82582bfbbba10cb0c9c0ea31230e771cd556511))
* **examples:** Introduce script to find valid combinations of seeds, samples, sizes and depths ([`954d204`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/954d204b7447cb79fbbcd12fd13f23472ffc75aa))
* **data_types:** Capture the index of the valid samples from the ZeMA dataset ([`c50a8f6`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/c50a8f68355036e3003c176a075df82c573da43c))

### Documentation
* **linear_inclusion notebook:** Update with most recent version of zema_emc_annotated ([`10e9b69`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/10e9b698e399ee03f1643ee6c58aa5b993519336))

## v0.4.0 (2023-01-07)
### Feature
* **uncertain_inputs:** Introduce direct value und uncertainties access for UncertainInputs ([`98a72aa`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/98a72aaceec8f43ffec1882cb97d3af87660f087))
* **generate_nn_params:** Introduce seeding into weight and bias generation ([`c4d2fd2`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/c4d2fd246d3c0fd011f083d12cabdb7a6e0f3388))
* **data_types:** Introduce index type for valid seed collections for ZeMA dataset ([`cf86be9`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/cf86be9be8cd6849241b048e56e5a2432572e5db))
* **activation_functions:** Introduce quadlu ([`a3c9947`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/a3c99474e7c722ebda2830588d62a5a4dba658f9))

### Documentation
* **README:** Update Roadmap and Getting started ([`d100a5b`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/d100a5ba4f9056091abf0dd48b1d6d84771eb56d))
* **linear_inclusion:** Introduce notebook to visualize and experiment with linear inclusion ([`256bb91`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/256bb91b85b3ddb4d82345db1988b1b83ef7d30d))
* **generate_nn_params:** Introduce module generate_nn_params into docs ([`decca7f`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/decca7ff53f4fe97667a5a5b517e0f3de7176d80))

## v0.3.0 (2022-12-28)
### Feature
* **optimize:** Introduce creation of deeper network parameters according to pytorch implementation ([`d4cf1f8`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/d4cf1f8038bf5fcd3ce112f57b7faf37d6336cc5))
* **generate_nn_params:** Introduce random generation of neural network parameters ([`dea1cc7`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/dea1cc749abf883c46161dc231c6798db39f8eb0))
* **optimize:** Introduce script to set up data and execute optimizer ([`961fc1a`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/961fc1a1a8253bf1da5d4ba70272ac46b06286fa))
* **pre_processing:** Introduce compute_values_label to determine an inputs label ([`d9bea2c`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/d9bea2c94b0c195966d3a052884697d7025ff9a9))
* **ilp:** Introduce linear optimization problem ([`628346f`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/628346f061236953cb672b4b66d4ff635309f185))

### Fix
* **pre_processing:** Fix bug in computation of z^(i)s and Theta^(i)s ([`b25f752`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/b25f7527ab15a3b6de278831f7a6bed8d377cbe1))
* **linear_program:** Set t to be unbounded from below ([`ca5785b`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/ca5785b033ae3c90f79bc383fd46af28c2779016))
* **linear_program:** Add objective to linear program ([`8d0682f`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/8d0682fe0154db514112583cc3750578f23afe63))
* **NNParams:** Fix default value creation and switch to tuple allowing differently sized layers ([`fdb002c`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/fdb002c049791a7503297bac28a0cc880615ea1b))
* **sigmoid:** Correct type hints ([`d70ae2c`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/d70ae2c01a9bda52626d9629132be298e1458112))
* **pre_processing:** Introduce required type cast and mypy ignore expression ([`1a9f1bb`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/1a9f1bbf1958cebef96426b6e22e5a23400f4ac7))

## v0.2.1 (2022-12-22)
### Fix
* **README:** Correct heading and description ([`d6a8d47`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/d6a8d4766ec870409b811710061851b3eddcf110))

## v0.2.0 (2022-12-22)
### Feature
* **data_types:** Introduce data_types module for more convenient coding and reading ([`38d2a8b`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/38d2a8bfd5aa243a5c2b99b73dd99d7d635a80c5))
* **activation_functions:** Introduce activation functions module ([`0063913`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/0063913bafe201914a797fb860065add70f23003))
* **pre_processing:** Introduce pre-processing module to prepare optimization ([`51e49c1`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/51e49c13f966425d0d75f71fcfcabc9aa41036b3))
* **uncertain_inputs:** Introduce data_acquisition's uncertain_inputs module to prepare data for optimization ([`5f98caa`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/5f98caa5b25bb59c5fd0bdb93c2ecea3fff53b33))

### Documentation
* **README:** Remove coverage badge as we cannot test and compute coverage on GitLab yet ([`486e3c0`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/486e3c056c31ae50727008522f6c1cfe43c7b349))
* **uncertain_inputs:** Introduce all pre_processing related modules into docs ([`bfdccea`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/bfdccea747b0469ca5fd0fff185bc1b960b2674a))

## v0.1.0 (2022-12-21)
### Feature
* Introduce first draft of package structure and docs ([`dcc4bd1`](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commit/dcc4bd18f76b2f8450e687b8c8fb73e4984e3354))
