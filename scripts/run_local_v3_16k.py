from __future__ import annotations

# Compatibility wrapper: V3 extends the proven local V2-16K runner with
# memory_v3 and Phase2 stages while preserving the original entrypoint.
from run_local_v2_16k_from_zero import main

if __name__ == "__main__":
    main()
