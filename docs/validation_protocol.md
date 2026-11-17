> Historical Phase 1 scope. For current setup and Phase 2/3 inference, see the phase-specific protocols and reproducibility guide.

# Prespecified validation protocol
1. Freeze split, estimator, transformation, primary metric, bootstrap seed, minimum sample size, improvement threshold, and rejection tolerance before viewing test findings.
2. Fit the baseline on training data only; challenger is a controlled sensitivity transform, not a retrained challenger.
3. Evaluate paired scores, probabilities, and binary decisions; save per-case outputs.
4. Bootstrap paired log-loss differences across independent cases only; report confidence bounds and effect size.
5. Apply safety veto first, case adequacy second, rejection and improvement gates next; always require human release authorization.
6. Archive environment metadata and the full output report. Review failures and unknowns before dissemination.
