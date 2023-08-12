## Bayesian Experimental Design for Computed Tomography with the Linearised Deep Image Prior

![DIP_experimental_design_intro](https://user-images.githubusercontent.com/50658913/172893131-0098c325-fe2a-4985-9644-5a4a6c04e1cc.png)

Paper: https://arxiv.org/abs/2207.05714 (presented at ICML Workshop on Adaptive Experimental Design and Active Learning in the Real World (ReALML) 2022, July 22, Baltimore, MD, USA)

Experimental results: [zenodo.org/record/6635902](https://zenodo.org/record/6635902) (includes tensorboard logs and other saved information).

In the following steps, we assume the code from this repository to checked out
as `$REPO`. For running the commands below, change the working directory by
`cd $REPO/src/experiments/`.


### 1.  Obtain initial DIP reconstruction and run MLL optimisation for linearised DIP prior hyperparameters

```shell
python bayes_dip.py use_double=True data=rectangles num_images=30 net.optim.gamma=3e-3 net.optim.iterations=19100 noise_specs.stddev=0.05 mrglik.optim.include_predcp=False mrglik.impl.vec_batch_size=25 mrglik.optim.iterations=500 mrglik.priors.clamp_variances=False beam_num_angle=200 angular_sub_sampling=40 mrglik.cg_impl.max_cg_iter=10 mrglik.impl.use_preconditioner=True
```

This step is required (once) for step 2a)-2d), but not for 2e) and 2f).


### 2.  Select new acquisition angles

![DIP_experimental_design](https://user-images.githubusercontent.com/50658913/172891797-a1be058b-9125-48fc-9b03-60f0817aca31.png)

Assume the output path of step 1 to be `$OUTPUT_BAYES_DIP`.

The following angle selection criteria are implemented:
  * expected squared error in measurement space (ESE); use with: `bed.use_EIG=False` (the default)
  * expected information gain (EIG); use with: `bed.use_EIG=True use_diagonal_EIG=False`
  * expected information gain assuming diagonality (diagonal EIG); use with: `bed.use_EIG=True use_diagonal_EIG=True`

The commands below use the ESE by default; to use EIG or diagonal EIG instead, append the respective options to the command.

#### 2a)  Selection by linearised DIP

```shell
python bayes_exp_design.py use_double=True data=rectangles num_images=30 mrglik.optim.include_predcp=False mrglik.impl.vec_batch_size=25 mrglik.priors.clamp_variances=False beam_num_angle=200 angular_sub_sampling=40 bed.mc_samples=3000 bed.reconstruct_every_k_step=5 bed.total_num_acq_projs=35 bed.acq_projs_batch_size=1 noise_specs.stddev=0.05 density.compute_single_predictive_cov_block.load_path=$OUTPUT_BAYES_DIP bed.hyperparam_path_baseline=$REPO/src/experiments/hyperparams/dip.yaml bed.use_objective_prior=False bed.update_network_params=False
```

This command also performs DIP reconstruction every 5 angles, so step 3a) can be skipped.

#### 2b)  Selection by linearised DIP, with DIP retrained every 5 angles

```shell
python bayes_exp_design.py use_double=True data=rectangles num_images=30 mrglik.optim.include_predcp=False mrglik.impl.vec_batch_size=25 mrglik.priors.clamp_variances=False beam_num_angle=200 angular_sub_sampling=40 bed.mc_samples=3000 bed.reconstruct_every_k_step=5 bed.total_num_acq_projs=35 bed.acq_projs_batch_size=1 noise_specs.stddev=0.05 density.compute_single_predictive_cov_block.load_path=$OUTPUT_BAYES_DIP bed.hyperparam_path_baseline=$REPO/src/experiments/hyperparams/dip_avg_best_iter.yaml bed.use_objective_prior=False bed.update_network_params=True
```

The DIP is retrained every 5 angles.  The retraining uses numbers of iterations
that were found to perform best on a validation set on average, listed in
`$REPO/src/experiments/hyperparams/dip_avg_best_iter.yaml`.  Some images require
more iterations to converge, therefore, to reach maximum PSNR (to be selected
post-hoc), the DIP reconstructions need to be recomputed with more iterations:
see step 3a).

#### 2c)  Selection by linearised DIP (g-prior)

```shell
python bayes_exp_design.py use_double=True data=rectangles num_images=30 mrglik.optim.include_predcp=False mrglik.impl.vec_batch_size=25 mrglik.priors.clamp_variances=False beam_num_angle=200 angular_sub_sampling=40 bed.mc_samples=3000 bed.reconstruct_every_k_step=5 bed.total_num_acq_projs=35 bed.acq_projs_batch_size=1 noise_specs.stddev=0.05 density.compute_single_predictive_cov_block.load_path=$OUTPUT_BAYES_DIP bed.hyperparam_path_baseline=$REPO/src/experiments/hyperparams/dip.yaml bed.use_objective_prior=True bed.update_scale_vec_via_refined_jac=False
```

This command also performs DIP reconstruction every 5 angles, so step 3a) can be skipped.

#### 2d)  Selection by linearised DIP (g-prior), with DIP retrained every 5 angles

```shell
python bayes_exp_design.py use_double=True data=rectangles num_images=30 mrglik.optim.include_predcp=False mrglik.impl.vec_batch_size=25 mrglik.priors.clamp_variances=False beam_num_angle=200 angular_sub_sampling=40 bed.mc_samples=3000 bed.reconstruct_every_k_step=5 bed.total_num_acq_projs=35 bed.acq_projs_batch_size=1 noise_specs.stddev=0.05 density.compute_single_predictive_cov_block.load_path=$OUTPUT_BAYES_DIP bed.hyperparam_path_baseline=$REPO/src/experiments/hyperparams/dip_avg_best_iter.yaml bed.use_objective_prior=True bed.update_scale_vec_via_refined_jac=True
```

The DIP is retrained every 5 angles.  The retraining uses numbers of iterations
that were found to perform best on a validation set on average, listed in
`$REPO/src/experiments/hyperparams/dip_avg_best_iter.yaml`.  Some images require
more iterations to converge, therefore, to reach maximum PSNR (to be selected
post-hoc), the DIP reconstructions need to be recomputed with more iterations:
see step 3a).

#### 2e)  Selection by isotropic model

```shell
python bayes_exp_design_linear_model.py use_double=True data=rectangles num_images=30 beam_num_angle=200 angular_sub_sampling=40 bed.reconstruct_every_k_step=5 bed.total_num_acq_projs=35 bed.acq_projs_batch_size=1 noise_specs.stddev=0.05 bed.linear_model_mll_optim.use_gp_model=False
```

#### 2f)  Selection by Matern-½ process

```shell
python use_double=True data=rectangles num_images=30 beam_num_angle=200 angular_sub_sampling=40 bed.reconstruct_every_k_step=5 bed.total_num_acq_projs=35 bed.acq_projs_batch_size=1 noise_specs.stddev=0.05 bed.linear_model_mll_optim.use_gp_model=True
```


### 3.  Reconstruct using different angle selections

![DIP_experimental_design_reco_performance](https://user-images.githubusercontent.com/50658913/172892786-042400a6-b8af-412a-ab8e-f5b9814d83ab.png)

Assume the output path of any angle selection step, i.e. one of 2a)-2f), to be
`$OUTPUT_ANGLE_SELECTION`.

#### 3a)  DIP reconstruction

#####  Selected angles from step 2.

```shell
python bayes_exp_design.py use_double=True data=rectangles num_images=30 mrglik.optim.include_predcp=False mrglik.impl.vec_batch_size=25 mrglik.priors.clamp_variances=False beam_num_angle=200 angular_sub_sampling=40 bed.mc_samples=3000 bed.reconstruct_every_k_step=5 bed.total_num_acq_projs=35 bed.acq_projs_batch_size=1 noise_specs.stddev=0.05 density.compute_single_predictive_cov_block.load_path=$OUTPUT_BAYES_DIP bed.hyperparam_path_baseline=$REPO/src/experiments/hyperparams/dip.yaml bed.use_best_inds_from_path=$OUTPUT_ANGLE_SELECTION
```

#####  Equidistant baseline

```shell
python bayes_exp_design.py use_double=True data=rectangles num_images=30 mrglik.optim.include_predcp=False mrglik.impl.vec_batch_size=25 mrglik.priors.clamp_variances=False beam_num_angle=200 angular_sub_sampling=40 bed.reconstruct_every_k_step=1 bed.total_num_acq_projs=35 bed.acq_projs_batch_size=5 noise_specs.stddev=0.05 density.compute_single_predictive_cov_block.load_path=$OUTPUT_BAYES_DIP bed.hyperparam_path_baseline=$REPO/src/experiments/hyperparams/dip.yaml bed.compute_equidistant_baseline=True
```

#### 3b)  TV reconstruction

#####  Selected angles from step 2.

```shell
python bayes_exp_design.py use_double=True data=rectangles num_images=30 mrglik.optim.include_predcp=False mrglik.impl.vec_batch_size=25 mrglik.priors.clamp_variances=False beam_num_angle=200 angular_sub_sampling=40 bed.mc_samples=3000 bed.reconstruct_every_k_step=5 bed.total_num_acq_projs=35 bed.acq_projs_batch_size=1 noise_specs.stddev=0.05 density.compute_single_predictive_cov_block.load_path=$OUTPUT_BAYES_DIP bed.use_alternative_recon='tvadam' bed.tvadam_hyperparam_path_baseline=$REPO/src/experiments/hyperparams/tvadam.yaml bed.use_best_inds_from_path=$OUTPUT_ANGLE_SELECTION
```

#####  Equidistant baseline

```shell
python bayes_exp_design.py use_double=True data=rectangles num_images=30 mrglik.optim.include_predcp=False mrglik.impl.vec_batch_size=25 mrglik.priors.clamp_variances=False beam_num_angle=200 angular_sub_sampling=40 bed.reconstruct_every_k_step=1 bed.total_num_acq_projs=35 bed.acq_projs_batch_size=5 noise_specs.stddev=0.05 density.compute_single_predictive_cov_block.load_path=$OUTPUT_BAYES_DIP bed.use_alternative_recon=tvadam bed.tvadam_hyperparam_path_baseline=$REPO/src/experiments/hyperparams/tvadam.yaml bed.compute_equidistant_baseline=True
```
