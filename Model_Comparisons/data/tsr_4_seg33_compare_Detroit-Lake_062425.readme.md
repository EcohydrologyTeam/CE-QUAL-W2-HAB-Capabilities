# CE-QUAL-W2 Model Comparisons

These CE-QUAL-W2 model comparisons were performed by Scott Hinz, LimnoTech, and delivered by email on June 24, 2025.

## CE-QUAL-W2 Versions

Detroit Lake inputs were simulated using three different W2 version 4.5 executables:

- Original: as provided with inputs to LimnoTech
- PSU: a July 2024 executable distributed by PSU that relates to the code to which LimnoTech made changes
- Limno: LimnoTech's revised version of the PSU code to add in some of the HAB-centric capabilities

## Goal

The goal was to verify if Detroit Lake model results could be reproduced with LimnoTech's modified code without turning on any of our new options. In each case, one-hour simulations take approximately 6 hours to compute, and they generate large amounts of output. Therefore, this verification focuses on Segment 33.


## Results

### PSU-compiled v4.5 vs LimnoTech's v4.5 with HAB capabilities

- These two versions yield yield similar results.
- Observed differences may be related to when the computed timesteps sharply decline. Potential timestep violations may be causing unexpected perturbations and slight shifts in when outputs are generated. This may then result in small differences in the output values.
- In general, these differences don't seem like a major issue.

### Original vs PSU-compiled and LimnoTech versions

- Larger prediction differences were observed between the "Original" executable and the other two versions, with relatively significant differences for some water quality parameters (DO, nutrients, etc.) showing up as early as Day 14 or so.
- This suggests that the Detroit Lake v4.5 model may have some code differences versus the PSU v4.5 model and LimnoTech's revised version of the PSU model.

## Questions

- Is more information available on the Original version of W2
- Is that code available to help us better understand why we would see larger than expected differences in the model predictions?
- For future comparisons, what specific model grid locations are of interest?
- How much should we focus on duplicating the "Original" Detroit Lake model predictions?

## Potential Solutions

- Recompile the base PSU code to verify if the issues are related to the compiler used

