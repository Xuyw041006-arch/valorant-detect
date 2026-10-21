# Final video event evaluation

The score uses all 130 raw events in `predictions/events.json`, with clip-local time mappings in `predictions/evidence.json`, and 155 AI-generated references in `final_video/truth.json`. The author performed a preliminary spot-check; these are uncertain development annotations, not independently verified ground truth.

Recorded results: TP=63, FP=67, FN=92, Precision=48.46%, Recall=40.65%. The reference/protocol SHA256 are recorded in `final_video/truth_freeze.json`; the external evaluation checked them before scoring. Validation code and logs are deliberately excluded from this public repository.

`final_video/protocol.json` is retained byte-for-byte with its original V10 / 132-record metadata. This is historical metadata only: the current V11 tracking output has 130 records, confirmed by the stored metrics. The annotation rules, semantic matching and ±0.5 s tolerance have not changed.

The 128 reviewed demonstration events are separate. Two confirmed false object detections were removed only from presentation; all original predictions, including errors, remain in scoring. GT/prediction skill stills are excluded. Agent, killer and weapon identities, detection boxes, trajectory IDs and minimap coordinates are not scored.
