# Circleworld Proper Attempt

## Scope
This was the first isolated Circleworld training attempt run as its own runtime lane, without modifying the Graduation restore path or the other diffusion/semantic ablations.

## Runtime
- Runtime: `circleworld_proto`
- Training entrypoint: [D:\RAFA\runtimes\circleworld_proto\train_circleworld.py](D:\RAFA\runtimes\circleworld_proto\train_circleworld.py)
- Base implementation: [D:\RAFA\lineages\04_positive_replacement\circleworld.py](D:\RAFA\lineages\04_positive_replacement\circleworld.py)

## Training Method
- method: cross-entropy parameter search
- target mode: `active_packets`
- train seeds: mixed `synthetic` and `naked_rafa`
- val seeds: mixed `synthetic` and `naked_rafa`
- recursion depth: `2`
- time steps: `96`

This is parameter training, not a learned neural head. It is meant to answer a narrower question:

Can the current Circleworld machinery be tuned into a beneficial structural operator before we add a richer learned lawpacket?

## Artifacts
- summary: [D:\RAFA\outputs\circleworld_proto\proper_attempt_2026-04-06\train_summary.json](D:\RAFA\outputs\circleworld_proto\proper_attempt_2026-04-06\train_summary.json)
- baseline report: [D:\RAFA\outputs\circleworld_proto\proper_attempt_2026-04-06\baseline_report.json](D:\RAFA\outputs\circleworld_proto\proper_attempt_2026-04-06\baseline_report.json)
- search history: [D:\RAFA\outputs\circleworld_proto\proper_attempt_2026-04-06\search_history.json](D:\RAFA\outputs\circleworld_proto\proper_attempt_2026-04-06\search_history.json)
- best config checkpoint: [D:\RAFA\checkpoints_circleworld_proto\proper_attempt_2026-04-06\circleworld_config_cem_v1.json](D:\RAFA\checkpoints_circleworld_proto\proper_attempt_2026-04-06\circleworld_config_cem_v1.json)

## Result

### Baseline active config on validation
- `mean_score = -0.23870471611618999`
- `mean_loss = 1.3733210802078246`
- `mean_major_gain = -0.001543603092432022`
- `mean_residue_drop = -0.0034054279327392577`
- `mean_promotability_gain = -0.001681380718946457`
- `mean_num_promotions = 3.6`

### Best trained active config on validation
- `mean_score = -0.22581976171582938`
- `mean_loss = 1.3365152597427368`
- `mean_major_gain = -5.771145224571228e-05`
- `mean_residue_drop = -9.592771530151368e-05`
- `mean_promotability_gain = -5.81599771976471e-05`
- `mean_num_promotions = 0.8`

### Best config
- `q_weights = [1.2199854781479629, 1.1182518884783725, 0.7026202445146327, 0.7262274362585671, 0.6039433364656246, 0.16780465518109403, 0.17007803808301536]`
- `promotion_threshold = 0.7859238822395697`
- `max_promotions = 3`
- `child_law_gain = 0.03586583092917991`
- `attack_window = 10`
- `persistence_momentum = 0.5178078915247202`

## Read
This run did improve the `active_packets` branch relative to its default config. The improvement came from making Circleworld much less aggressive:
- higher promotion threshold
- much smaller child-law gain
- fewer promotions
- lower weight on the high-q tail

That means the current machinery is learning that most interventions are harmful.

## Interpretation
This is useful, but not the victory condition.

What we have shown:
- the Circleworld lane can be trained in isolation
- the tuned config is more stable than the default active-law config
- the current active-law implementation is too blunt, and the best optimizer response is mostly to suppress it

What we have not shown:
- a true improvement over `no_promotion`
- robust positive gains on `naked_rafa` seeds
- a learned local law that beats passive restraint

## Verdict
Circleworld is now an actual trainable runtime lane, but its first proper attempt says the same thing the ablation said in softer language:

The current packet-to-law mechanism is not yet a good law. The branch improves when it behaves more conservatively, not when it acts more intelligently.
