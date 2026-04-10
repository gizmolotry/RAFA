# Shared Battlefield Continuity (2026-04-10)

This report puts Graduation RAFA and Circleworld RAFA on the same continuity battlefield. The horizons differ: Graduation is measured on 2-second canonical clips, Circleworld on 10-second anchor-conditioned renders.

## Graduation RAFA
- mean_loop_autocorr_peak: original=0.345864, restore=0.357628, delta=+0.011763
- mean_loop_period_seconds: original=0.288000, restore=0.336000, delta=+0.048000
- mean_adjacent_chunk_similarity: original=0.891579, restore=0.877137, delta=-0.014442
- mean_nonlocal_chunk_repeat: original=0.909945, restore=0.907553, delta=-0.002392
- mean_first_chunk_reentry: original=0.892443, restore=0.878265, delta=-0.014178

Read: the restore candidate is close enough to the original pack to treat as the live engineering baseline, though it is not bit-exact.

## Circleworld RAFA: band vs reference
- mean_loop_autocorr_peak: reference=0.571913, output=0.571491, delta=-0.000422
- mean_loop_period_seconds: reference=1.830400, output=1.830400, delta=+0.000000
- mean_adjacent_chunk_similarity: reference=0.882676, output=0.881955, delta=-0.000720
- mean_nonlocal_chunk_repeat: reference=0.963190, output=0.962398, delta=-0.000792
- mean_first_chunk_reentry: reference=0.879727, output=0.878840, delta=-0.000887

## Circleworld RAFA: constrained vs reference
- mean_loop_autocorr_peak: reference=0.626590, output=0.626455, delta=-0.000135
- mean_loop_period_seconds: reference=1.458667, output=1.458667, delta=+0.000000
- mean_adjacent_chunk_similarity: reference=0.905511, output=0.904923, delta=-0.000588
- mean_nonlocal_chunk_repeat: reference=0.961447, output=0.961072, delta=-0.000375
- mean_first_chunk_reentry: reference=0.904177, output=0.903648, delta=-0.000529

## Interpretation
- Graduation RAFA currently owns the sound-engine lane: it has a certified local-audio baseline and a recoverable runtime.
- Circleworld band/constrained do not yet create a new macro-time law. Their continuity signatures are almost the same as their anchors.
- So the current best Circleworld runs are primarily real-audio-preserving interventions, not long-horizon scene builders.
- The next real research question is not "does Circleworld sound okay?" but "can Circleworld reduce loop/re-entry while remaining on the real-audio frontier?"
