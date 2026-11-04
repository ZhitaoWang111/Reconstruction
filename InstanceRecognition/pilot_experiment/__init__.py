"""Build one Harness window input, then ask several models for one turn each.

This is a *pilot*, not a pipeline. The question it answers is narrow: given the
exact input the Harness would send, how long does each candidate model take and
what does it do with that one turn? Nothing is committed, no state is advanced,
and the window input is built once and reused byte-for-byte so the comparison is
between models rather than between inputs.

Reusing `build_window_content` is the point. A hand-written prompt would measure
a model's response to something the Harness never sends.

    python -m pilot_experiment.run_pilot build          # build the window input
    python -m pilot_experiment.run_pilot run --models ...  # one turn per model
"""
