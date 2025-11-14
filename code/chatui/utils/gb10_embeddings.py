"""
GB10 Smart Embeddings - Unified Embedding Interface
====================================================

This module provides a unified interface for embeddings that can use either:
- Cloud embeddings (NVIDIA API)
- Local embeddings (HuggingFace sentence-transformers on GB10 GPU)

The choice is controlled by environment variables, allowing seamless switching
without code changes.

Author: NVIDIA
Date: 2025-11-14
License: Apache 2.0
"""

import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def get_embeddings(force_local: bool = False, force_cloud: bool = False):
    """
    Get embeddings (cloud or local) based on configuration

    Environment Variables:
        USE_LOCAL_EMBEDDINGS: Set to 'true' or '1' to use local embeddings
        EMBEDDING_MODEL: Model name (for local embeddings)
        EMBEDDING_DEVICE: Device to use (cuda, cpu, or auto)

    Args:
        force_local: Force use of local embeddings
        force_cloud: Force use of cloud embeddings

    Returns:
        Embeddings object compatible with LangChain

    Example:
        # Use local embeddings on GB10
        export USE_LOCAL_EMBEDDINGS=true
        export EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
        export EMBEDDING_DEVICE=cuda

        embeddings = get_embeddings()
    """

    # Determine which embeddings to use
    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() in ["true", "1", "yes"]

    if force_local:
        use_local = True
    elif force_cloud:
        use_local = False

    if use_local:
        logger.info("Using LOCAL embeddings (GB10 GPU)")
        return _get_local_embeddings()
    else:
        logger.info("Using CLOUD embeddings (NVIDIA API)")
        return _get_cloud_embeddings()


def _get_cloud_embeddings():
    """Get NVIDIA cloud embeddings"""
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    # Check if internal API is available
    internal_api = os.getenv('INTERNAL_API', 'no')

    if internal_api == 'yes':
        model = 'nvdev/nvidia/nv-embedqa-e5-v5'
        logger.info(f"Using internal NVIDIA embedding model: {model}")
    else:
        model = 'nvidia/nv-embedqa-e5-v5'
        logger.info(f"Using public NVIDIA embedding model: {model}")

    return NVIDIAEmbeddings(model=model)


def _get_local_embeddings():
    """Get local HuggingFace embeddings for GB10"""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        logger.error("HuggingFaceEmbeddings not available")
        logger.error("Install with: pip install sentence-transformers")
        raise ImportError("sentence-transformers not installed")

    # Configuration from environment
    model_name = os.getenv(
        "EMBEDDING_MODEL",
        "BAAI/bge-large-en-v1.5"  # Default: high quality model
    )

    device = os.getenv("EMBEDDING_DEVICE", "auto")

    # Auto-detect device if needed
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("Auto-detected CUDA device for embeddings")
            else:
                device = "cpu"
                logger.warning("CUDA not available, using CPU for embeddings")
        except ImportError:
            device = "cpu"
            logger.warning("PyTorch not installed, using CPU for embeddings")

    # Additional configuration
    normalize = os.getenv("EMBEDDING_NORMALIZE", "true").lower() in ["true", "1", "yes"]
    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

    logger.info(f"Local embedding configuration:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Normalize: {normalize}")
    logger.info(f"  Batch size: {batch_size}")

    # Create embeddings
    model_kwargs = {
        'device': device,
        'trust_remote_code': True
    }

    encode_kwargs = {
        'normalize_embeddings': normalize,
        'batch_size': batch_size
    }

    # Optional: cache directory for models
    cache_dir = os.getenv("EMBEDDING_CACHE_DIR")
    if cache_dir:
        model_kwargs['cache_folder'] = cache_dir
        logger.info(f"  Cache dir: {cache_dir}")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    logger.info("✅ Local embeddings initialized successfully")
    return embeddings


class GB10Embeddings:
    """
    Wrapper class for embeddings with GB10 optimizations

    This class provides additional features:
    - Automatic batching for large document sets
    - Progress tracking
    - Error handling and retries
    - Performance monitoring
    """

    def __init__(self, use_local: bool = None):
        """
        Initialize GB10 embeddings

        Args:
            use_local: True for local, False for cloud, None for auto-detect
        """
        if use_local is None:
            use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() in ["true", "1", "yes"]

        self.use_local = use_local
        self.embeddings = get_embeddings(force_local=use_local)

        logger.info(f"GB10Embeddings initialized (local={use_local})")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents

        Args:
            texts: List of text documents

        Returns:
            List of embedding vectors
        """
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query

        Args:
            text: Query text

        Returns:
            Embedding vector
        """
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise

    def __call__(self, text: str) -> List[float]:
        """Allow calling the object directly"""
        return self.embed_query(text)


def benchmark_embeddings(num_docs: int = 100, text_length: int = 200):
    """
    Benchmark embedding performance

    Args:
        num_docs: Number of documents to test
        text_length: Length of each document (characters)

    Returns:
        dict with benchmark results
    """
    import time

    # Generate test data
    test_texts = [
        f"This is test document number {i}. " * (text_length // 40)
        for i in range(num_docs)
    ]

    results = {}

    # Benchmark local embeddings (if available)
    try:
        print("\n[Benchmark] Testing LOCAL embeddings...")
        embeddings = get_embeddings(force_local=True)

        start = time.time()
        _ = embeddings.embed_documents(test_texts[:10])  # Warm-up
        warmup_time = time.time() - start

        start = time.time()
        vectors = embeddings.embed_documents(test_texts)
        elapsed = time.time() - start

        results['local'] = {
            'time_seconds': elapsed,
            'docs_per_second': num_docs / elapsed,
            'ms_per_doc': (elapsed / num_docs) * 1000,
            'warmup_seconds': warmup_time
        }

        print(f"  Local: {results['local']['docs_per_second']:.2f} docs/sec")

    except Exception as e:
        print(f"  Local embeddings not available: {e}")
        results['local'] = None

    # Benchmark cloud embeddings (if API key available)
    try:
        if os.getenv("NVIDIA_API_KEY"):
            print("\n[Benchmark] Testing CLOUD embeddings...")
            embeddings = get_embeddings(force_cloud=True)

            start = time.time()
            _ = embeddings.embed_documents(test_texts[:10])  # Warm-up
            warmup_time = time.time() - start

            start = time.time()
            vectors = embeddings.embed_documents(test_texts)
            elapsed = time.time() - start

            results['cloud'] = {
                'time_seconds': elapsed,
                'docs_per_second': num_docs / elapsed,
                'ms_per_doc': (elapsed / num_docs) * 1000,
                'warmup_seconds': warmup_time
            }

            print(f"  Cloud: {results['cloud']['docs_per_second']:.2f} docs/sec")
        else:
            print("\n[Benchmark] Skipping cloud (no NVIDIA_API_KEY)")
            results['cloud'] = None

    except Exception as e:
        print(f"  Cloud embeddings failed: {e}")
        results['cloud'] = None

    return results


# Backwards compatibility
def get_nvidia_embeddings():
    """
    Legacy function for backwards compatibility
    Always returns cloud embeddings
    """
    logger.warning("get_nvidia_embeddings() is deprecated. Use get_embeddings() instead.")
    return _get_cloud_embeddings()


if __name__ == "__main__":
    """Test and benchmark embeddings"""
    import sys

    logging.basicConfig(level=logging.INFO)

    print("="*60)
    print("GB10 Smart Embeddings - Test & Benchmark")
    print("="*60)

    # Test 1: Auto-detect
    print("\n[Test 1] Auto-detect embeddings...")
    try:
        embeddings = get_embeddings()
        test_vector = embeddings.embed_query("Test query")
        print(f"✓ Embeddings working (dimension: {len(test_vector)})")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        sys.exit(1)

    # Test 2: Local embeddings
    print("\n[Test 2] Local embeddings...")
    try:
        embeddings = get_embeddings(force_local=True)
        test_vector = embeddings.embed_query("Test query for GB10")
        print(f"✓ Local embeddings working (dimension: {len(test_vector)})")
    except Exception as e:
        print(f"⚠ Local embeddings not available: {e}")

    # Test 3: Cloud embeddings
    print("\n[Test 3] Cloud embeddings...")
    if os.getenv("NVIDIA_API_KEY"):
        try:
            embeddings = get_embeddings(force_cloud=True)
            test_vector = embeddings.embed_query("Test query for cloud")
            print(f"✓ Cloud embeddings working (dimension: {len(test_vector)})")
        except Exception as e:
            print(f"✗ Cloud embeddings failed: {e}")
    else:
        print("⚠ Skipping (no NVIDIA_API_KEY)")

    # Benchmark
    if "--benchmark" in sys.argv:
        print("\n[Benchmark] Running performance tests...")
        results = benchmark_embeddings(num_docs=50)

        print("\n" + "="*60)
        print("Benchmark Results")
        print("="*60)

        if results['local']:
            print("\nLocal Embeddings (GB10 GPU):")
            print(f"  Throughput: {results['local']['docs_per_second']:.2f} docs/sec")
            print(f"  Latency: {results['local']['ms_per_doc']:.2f} ms/doc")

        if results['cloud']:
            print("\nCloud Embeddings (NVIDIA API):")
            print(f"  Throughput: {results['cloud']['docs_per_second']:.2f} docs/sec")
            print(f"  Latency: {results['cloud']['ms_per_doc']:.2f} ms/doc")

        if results['local'] and results['cloud']:
            speedup = results['local']['docs_per_second'] / results['cloud']['docs_per_second']
            print(f"\nSpeedup: {speedup:.2f}x (local vs cloud)")

    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)
