## 2026-09-26 - Split-stage cache reservations

**Context**: A split stage advertised a context that one session could fit, but multiple long requests could together exceed its cache budget after prompt execution had started.

**Learning**: The budget must account for the capacity that a growable key/value buffer may allocate, not only the positions already filled. A stage can make that decision before receiving prompt tokens. The existing request lease, close operation, and checkpoint rewind provide the three release points.

**Pattern**: Reserve the full requested prompt-plus-completion capacity at both stages, charge each active session for rounded buffer growth, and report aggregate-capacity rejection as overload. After rewind, charge a saved checkpoint for its actual retained allocation.

**Anti-pattern**: Do not use the model's format context or a single-session cache limit as proof that simultaneous stage sessions fit in device memory.
