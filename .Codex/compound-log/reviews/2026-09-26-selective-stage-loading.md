# 2026-09-26 - Selective stage loading review

## Delivered

- A GGUF loader for one proper prefix or suffix range, including only that stage's endpoint tensors.
- Stage-owned CPU sessions with independent key/value caches and position checks.
- A full-file digest and complementary-range check for stage pairing.

## Validation

The Q4_K_M test compared four token positions against the full model and found exactly matching scores. It checked that each stage stores fewer weight bytes than the complete model, rejected invalid ranges and a different GGUF variant, and kept cache positions stable after invalid hidden inputs. The test observed 99,230,976 stored weight bytes for the full model, 59,346,432 for the prefix, and 59,348,736 for the suffix.

## Remaining work

Both stages still run in one process during the parity test. There is no activation wire protocol, paired-device stage placement, stage-specific admission, or resident-memory measurement yet. A suffix failure requires the split request to discard both stage sessions because the prefix may already have advanced.
