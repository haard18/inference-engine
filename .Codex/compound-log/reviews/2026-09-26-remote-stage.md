# 2026-09-26 - Remote stage review

## Delivered

- A suffix-stage peer service using the existing approved-certificate mutual TLS configuration.
- A private child-process bridge with bounded activation and score bodies, capacity reporting, a bounded queue, a per-step deadline, and session close.
- A peer client method that sends one activation without replay and validates returned scores.
- A server command that launches the suffix endpoint using the owner's paired device state.

## Validation

The real Q4_K_M test launched the suffix service from the command line, sent four encrypted stage requests from an approved device, and matched full-model scores exactly. It rejected an unapproved device and a wrong request ID. Stopping the service made the next stage request fail without replay. A separate cancellation test held a fake stage calculation, canceled its caller, and confirmed that the next request used a fresh child. Strict Clippy, formatting, and all 54 release tests passed.

## Remaining work

The normal chat API still executes each request on a complete model. It needs an opt-in split placement that admits both stages, shares one deadline, returns streamed tokens, closes stage sessions, and reports stage or link loss. Two physical Mac measurements and load tests remain open.
