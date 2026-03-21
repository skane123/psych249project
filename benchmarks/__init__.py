BENCHMARK_REGISTRY = {}

# Import Benchmark to trigger registrations
# Imports are wrapped in try/except so that a missing optional dependency
# (e.g. decord on Apple Silicon) only disables the affected benchmarks
# rather than crashing the entire registry.
from benchmarks.NSD import NSDSharedBenchmark

try:
    from benchmarks.BMD import BMDBenchmark
except ImportError as _e:
    print(f"[Warning] Failed to import benchmark module: BMD\n{_e}")

from benchmarks.TVSD import TVSDBenchmark

try:
    from benchmarks.Physion import PhysionContact
except ImportError as _e:
    print(f"[Warning] Failed to import benchmark module: Physion.Contact\n{_e}")

try:
    from benchmarks.Physion import PhysionPlacement
except ImportError as _e:
    print(f"[Warning] Failed to import benchmark module: Physion.Placement\n{_e}")

from benchmarks.SSV2 import SSV2Benchmark, ModifiedSSV2Benchmark
from benchmarks.V1SineGratings import V1SineGratingsBenchmark
from benchmarks.LeBel2023 import LeBel2023Benchmark
