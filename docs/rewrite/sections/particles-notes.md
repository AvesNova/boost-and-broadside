# Notes: K-particle position heads without collapse (survey 2026-11-07)

From a reader agent (abstracts, method sections or code checked; [unverified] marked).
The problem: per hidden enemy, K ~ 4 particles (x, y, sigma, weight logit) relative to the
map centre (the super token's frame), on a wrapping map, fed back next tick and turned into
rotary phasors sum_k w_k exp(i w x_k) exp(-sigma_k^2 w^2 / 2). One true position per sample:
multimodality is learned across samples.

## Methods

- **MDN** (Bishop 1994): GMM NLL. Unstable, initialisation-sensitive, collapses to one mode
  (Makansi et al. CVPR 2019, arXiv 1906.03631: EWTA, a shrinking top-k WTA, then a mixture
  fit). Gaussian NLL converges early to poor fits (Seitzer et al. ICLR 2022, arXiv
  2203.09168: beta-NLL, scale each sample's loss by stop-grad sigma^(2 beta), beta 0.5).
- **WTA / MCL** (Guzman-Rivera 2012; Lee 2016 arXiv 1606.07839): the closest hypothesis
  alone regresses; optimum a Voronoi tessellation; dead hypotheses. Relaxed WTA (Rupprecht
  ICCV 2017, arXiv 1612.00197): losers get eps/(K-1); biases toward the barycentre.
  rMCL (Letzelter NeurIPS 2023, arXiv 2311.01052): scoring heads for each hypothesis's win
  probability. Annealed WTA (Perera NeurIPS 2024, arXiv 2407.15580): loss
  sum_k softmin_T(l)_k l_k, weights detached, T cooled (T0 0.5, x0.95 per epoch); above
  T_c = 2 lambda_max(Cov) all hypotheses sit at the barycentre, then split in stages; lower
  seed variance than MCL. Collapse diagnostic: entropy of the winner-count histogram.
- **Trajectory forecasting** (MultiPath++ arXiv 2111.14973; Wayformer arXiv 2207.05844;
  MTR NeurIPS 2022 arXiv 2209.13508; QCNet CVPR 2023): hard-assign the truth to one mode,
  NLL (Gaussian or Laplace) on that mode, cross-entropy or mixture NLL for the weights.
  MTR: assigning by the predicted trajectory (not fixed intention points) hurt as K grew,
  the positive mode "switching randomly" between queries: relevant to a recurrent model.
  Diffusion/flow forecasters avoid collapse by sampling; too costly per tick.
- **Wrapped domains:** von Mises mixtures (Bishop & Legleye 1994; Prokudin 2018 arXiv
  1805.03430). exp(i w x) exp(-sigma^2 w^2 / 2) is exactly the Fourier coefficient of a
  wrapped normal, so a wrapped-Gaussian mixture matches the phasors we build.
- **Recurrent particles:** PF-RNN (Ma et al. AAAI 2020, arXiv 1905.12885), soft resampling
  (mix with uniform) keeps gradients alive; survey Chen & Li 2023 (arXiv 2302.09639).

## The reader's recommendation (K = 4, per hidden ship)

1. Means: annealed-WTA regression, softmin over the wrapped (minimum-image) squared
   distance, T cooled from ~2 lambda_max of the position spread to a small floor (not 0;
   or relaxed eps ~ 0.05) as insurance against dead particles under RL's moving data.
2. Weights: cross-entropy on the soft winner (rMCL-style), or a mixture NLL with means and
   sigmas detached (QCNet-style).
3. Sigmas: each particle's wrapped-Gaussian NLL weighted by its responsibility, a sigma
   floor, beta-NLL (beta ~ 0.5); a 3 x 3 image sum while sigma << map size.
4. Feedback per slot (particle k at t+1 conditioned on its own previous state) for
   persistent identity, avoiding MTR's switching; break symmetry at spawn or reveal with
   small fixed offsets; detach fed-back particles at first.
5. Monitor winner-count entropy, sigma calibration (coverage), weight reliability.

What would change it: winners concentrate (unimodal) -> K 1-2; negative evidence ("not in
my view") makes crescents that isotropic components fit poorly -> anisotropic or larger K;
sigma near the map size -> full wrapped normal or von Mises.
