MV3 completed a reproducible frozen DINOv3/Primus baseline across12 sources. This independent MV4 change runs the released official MitoNet_v1 on exactly the same38 development regions, retaining the six existing DINOv3 runs and all228 metrics. It answers how competitive the complete systems are; it does not equalize their original supervision or claim image-isolated generalization.

MitoNet uses the official empanada 2D batch workflow, fixed scale/thresholds, official tiling and instance postprocessing. A headless adapter compiles unchanged hash-locked upstream Engine2d/Preprocessor definitions. GT is read only for scoring with the existing evaluator; Han retains whole-GT evaluation. Native per-plane panoptic and instance arrays remain on the server. GPU0 is prohibited, the website/history are protected, and extra dependencies live in a separate MV4 environment.

Initial delivery includes code, configuration, official asset metadata and per-turn Markdown/handoff. Scientific execution is pending validated weights and smoke; no MitoNet score is claimed at this point. CPU checks:21 tests plus repository/registry validation. Final results and actual environment/deployment receipts will be added after completion. Stop after the report; no automatic follow-on experiment.

This PR targets the existing MV3 branch because PR2 is still an unmerged draft. No automatic merge.
