# Nested Commitment Retrospective (2026-04-11)

This report compares the three main Circleworld regimes on the new fork/resume assay: preservation (`penalized`), balanced real-audio (`constrained`), and structural winner (`full`).

## full
- overall_read: `nested_commitment_not_yet_established`
- airplane_takeoff: verdict=`over_rigid_attractor`, mean_coarse_env_corr=0.9997, mean_meso_chunk_similarity=0.9997, mean_fine_texture_distance=0.0052, max_fine_texture_distance=0.0167, packet_noop=True
- sax_like_clarinet: verdict=`over_rigid_attractor`, mean_coarse_env_corr=0.9970, mean_meso_chunk_similarity=0.9830, mean_fine_texture_distance=0.0073, max_fine_texture_distance=0.0289, packet_noop=True

## constrained
- overall_read: `nested_commitment_not_yet_established`
- airplane_takeoff: verdict=`over_rigid_attractor`, mean_coarse_env_corr=1.0000, mean_meso_chunk_similarity=0.9998, mean_fine_texture_distance=0.0057, max_fine_texture_distance=0.0171, packet_noop=True
- sax_like_clarinet: verdict=`over_rigid_attractor`, mean_coarse_env_corr=0.9964, mean_meso_chunk_similarity=0.9813, mean_fine_texture_distance=0.0081, max_fine_texture_distance=0.0305, packet_noop=True

## penalized
- overall_read: `nested_commitment_not_yet_established`
- airplane_takeoff: verdict=`over_rigid_attractor`, mean_coarse_env_corr=0.9999, mean_meso_chunk_similarity=0.9995, mean_fine_texture_distance=0.0085, max_fine_texture_distance=0.0267, packet_noop=True
- sax_like_clarinet: verdict=`over_rigid_attractor`, mean_coarse_env_corr=0.9843, mean_meso_chunk_similarity=0.9712, mean_fine_texture_distance=0.0154, max_fine_texture_distance=0.0482, packet_noop=True

## Read
- packet perturbations are effectively no-ops across all three regimes
- fine noise changes texture a little, but does not create sibling continuations with distinct meso development
- promotion perturbations mostly leave the world unchanged rather than creating lawful branch siblings
- implication: The current recursion appears to preserve or freeze an attractor rather than refine a coarse world-law into differentiated child continuations.
- next move: Split internal state into coarse world-state, meso development-state, and fine detail-state before training continuity-aware objectives.
