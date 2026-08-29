#!/usr/bin/env python3
"""Train-only percentile candidates. Wrapper around CLI analyze-normalizer-percentiles."""

from microscopy_vae.cli import main

if __name__ == "__main__":
    import sys

    sys.argv = ["microscopy-vae", "analyze-normalizer-percentiles", *sys.argv[1:]]
    main()
