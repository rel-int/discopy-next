> Fix #442: raise ValueError on compressed fix/power_fix with max_chi=1 when bond > 1

- [x] Raise ValueError naming max_chi >= 2 in fix and power_fix when max_chi == 1 and needed > 1, tested on real QuimbBackend
- [WIP] @agy-442-2026-11-26 12:50 Add CHANGELOG.md entry
