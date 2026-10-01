"""
Backward compatibility wrapper for ImageNetValBenchmark.
Aliased to benchmarks.imagenet_val.ImageNetValBenchmark.
"""

from benchmarks.imagenet_val import ImageNetValBenchmark

# Clean backward-compatible alias and subclass
class ImageNetValBenchmarkTTA(ImageNetValBenchmark):
    pass
