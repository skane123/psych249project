from benchmarks.BBS import BenchmarkScore
from benchmarks.BBS_online import OnlineBenchmarkScore
from data.modified_SSV2 import (
    SSV2PrunedStimulusTrainSet,
    AugmentedSSV2PrunedStimulusTrainSet,
    SSV2PrunedStimulusTestSet
)
from benchmarks import BENCHMARK_REGISTRY


class ModifiedSSV2Benchmark(OnlineBenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=(SSV2PrunedStimulusTrainSet,
                                  SSV2PrunedStimulusTrainSet),
            stimulus_test_class=SSV2PrunedStimulusTestSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            num_classes=40,
            dataloader_batch_size=batch_size,
            num_workers=16,
            debug=debug,
        )


BENCHMARK_REGISTRY["ModifiedSSV2Benchmark"] = ModifiedSSV2Benchmark


class ModifiedAugmentedSSV2Benchmark(OnlineBenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=(
                AugmentedSSV2PrunedStimulusTrainSet, SSV2PrunedStimulusTrainSet),
            stimulus_test_class=SSV2PrunedStimulusTestSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            num_classes=40,
            dataloader_batch_size=batch_size,
            num_workers=16,
            debug=debug,
        )


BENCHMARK_REGISTRY["ModifiedAugmentedSSV2Benchmark"] = ModifiedAugmentedSSV2Benchmark

