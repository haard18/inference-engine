## 2026-09-26 - Engine and serving boundary

**Context:** We defined a local AI serving system that can later pool two Macs.

**Learning:** Device pooling and request routing do not make a system an inference engine. The engine must execute the model itself.

**Pattern:** Prove correct inference on one device before adding the serving API or coordinator.

**Anti-pattern:** Building a coordinator around another runtime while describing it as our own inference engine.
