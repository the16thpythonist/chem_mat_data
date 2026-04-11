"""
Benchmark tests for streaming dataset functionality.

These tests benchmark the performance of GraphDataset with different numbers of workers.
They are skipped by default and must be explicitly enabled.

To run these tests:
    pytest tests/test_dataset_benchmark.py -m benchmark -v

Or to run all tests including benchmarks:
    pytest tests/test_dataset_benchmark.py -v
"""

import os
import time
import tempfile
import pytest
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from typing import Tuple

from chem_mat_data.dataset import GraphDataset
from chem_mat_data.main import load_smiles_dataset
from chem_mat_data.processing import MoleculeProcessing
from tests.utils import ARTIFACTS_PATH


# Custom marker for benchmark tests
pytestmark = pytest.mark.benchmark


@pytest.mark.benchmark
class TestGraphDatasetBenchmark:
    """Benchmark tests for GraphDataset with different worker counts."""

    DATASET_NAME = 'esol'  # ~1100 molecules, good size for benchmarking
    WORKER_COUNTS = [0, 1, 2, 4, 8]  # Different worker configurations to test
    NUM_MOLECULES_TO_PROCESS = 100  # Process subset for faster testing

    @pytest.fixture
    def temp_folder(self):
        """Provide a temporary folder for dataset storage."""
        with tempfile.TemporaryDirectory() as temp_path:
            yield temp_path

    def _benchmark_worker_count(
        self, num_workers: int, temp_folder: str, num_molecules: int = None
    ) -> Tuple[float, int]:
        """
        Benchmark graph processing with specific worker count.

        :param num_workers: Number of parallel workers (0 = sequential)
        :param temp_folder: Temporary folder for dataset storage
        :param num_molecules: Number of molecules to process (None = all)

        :returns: Tuple of (elapsed_time, molecules_processed)
        """
        dataset = GraphDataset(
            dataset=self.DATASET_NAME,
            num_workers=num_workers,
            folder_path=temp_folder,
            buffer_size=100,
        )

        molecules_processed = 0
        start_time = time.time()

        for i, graph in enumerate(dataset):
            molecules_processed += 1
            if num_molecules and i + 1 >= num_molecules:
                break

        elapsed = time.time() - start_time

        # Cleanup
        dataset.close()

        return elapsed, molecules_processed

    @pytest.mark.skip(reason='Benchmark test - takes too long for regular unit tests')
    def test_benchmark_single_run(self, temp_folder):
        """
        Single benchmark run to verify the benchmark infrastructure works.
        """
        elapsed, count = self._benchmark_worker_count(
            num_workers=2, temp_folder=temp_folder, num_molecules=10
        )

        assert elapsed > 0
        assert count == 10
        print(f'\nBenchmark test: Processed {count} molecules in {elapsed:.3f}s')

    @pytest.mark.skip(reason='Benchmark test - takes too long for regular unit tests')
    def test_benchmark_worker_comparison(self, temp_folder):
        """
        Benchmark GraphDataset with different worker counts.

        This test processes a fixed number of molecules with different worker
        configurations and compares the timing results.
        """
        print(f'\n{"=" * 70}')
        print(f'Benchmarking GraphDataset with {self.DATASET_NAME} dataset')
        print(f'Processing {self.NUM_MOLECULES_TO_PROCESS} molecules')
        print(f'{"=" * 70}\n')

        results = []

        for num_workers in self.WORKER_COUNTS:
            print(f'Testing with {num_workers} worker(s)...', end=' ', flush=True)

            elapsed, count = self._benchmark_worker_count(
                num_workers=num_workers,
                temp_folder=temp_folder,
                num_molecules=self.NUM_MOLECULES_TO_PROCESS,
            )

            throughput = count / elapsed  # molecules per second
            results.append(
                {
                    'num_workers': num_workers,
                    'elapsed': elapsed,
                    'count': count,
                    'throughput': throughput,
                }
            )

            print(f'{elapsed:.3f}s ({throughput:.1f} mol/s)')

        # Print summary table
        print(f'\n{"=" * 70}')
        print('Summary:')
        print(f'{"=" * 70}')
        print(f'{"Workers":>10} {"Time (s)":>12} {"Throughput":>15} {"Speedup":>12}')
        print(f'{"-" * 70}')

        baseline_time = results[0]['elapsed']  # Sequential time as baseline

        for result in results:
            speedup = baseline_time / result['elapsed']
            print(
                f'{result["num_workers"]:>10d} '
                f'{result["elapsed"]:>12.3f} '
                f'{result["throughput"]:>12.1f} mol/s '
                f'{speedup:>11.2f}x'
            )

        print(f'{"=" * 70}\n')

        # Store results for plotting test
        self._benchmark_results = results

        # Basic assertions
        assert all(r['count'] == self.NUM_MOLECULES_TO_PROCESS for r in results)
        assert all(r['elapsed'] > 0 for r in results)

    @pytest.mark.skip(reason='Benchmark test - takes too long for regular unit tests')
    def test_benchmark_scaling_efficiency(self, temp_folder):
        """
        Test scaling efficiency of parallel processing.

        Computes and reports parallel efficiency for different worker counts.
        """
        print(f'\n{"=" * 70}')
        print('Parallel Scaling Efficiency Test')
        print(f'{"=" * 70}\n')

        worker_counts = [1, 2, 4, 8]
        results = []

        # Get baseline (1 worker)
        print('Measuring baseline (1 worker)...', end=' ', flush=True)
        baseline_elapsed, baseline_count = self._benchmark_worker_count(
            num_workers=1,
            temp_folder=temp_folder,
            num_molecules=self.NUM_MOLECULES_TO_PROCESS,
        )
        print(f'{baseline_elapsed:.3f}s')

        results.append(
            {
                'num_workers': 1,
                'elapsed': baseline_elapsed,
                'speedup': 1.0,
                'efficiency': 1.0,
            }
        )

        # Test other worker counts
        for num_workers in worker_counts[1:]:
            print(f'Testing {num_workers} workers...', end=' ', flush=True)

            elapsed, count = self._benchmark_worker_count(
                num_workers=num_workers,
                temp_folder=temp_folder,
                num_molecules=self.NUM_MOLECULES_TO_PROCESS,
            )

            speedup = baseline_elapsed / elapsed
            efficiency = speedup / num_workers  # Ideal efficiency is 1.0

            results.append(
                {
                    'num_workers': num_workers,
                    'elapsed': elapsed,
                    'speedup': speedup,
                    'efficiency': efficiency,
                }
            )

            print(f'{elapsed:.3f}s (efficiency: {efficiency:.1%})')

        # Print summary
        print(f'\n{"=" * 70}')
        print('Scaling Efficiency Summary:')
        print(f'{"=" * 70}')
        print(f'{"Workers":>10} {"Speedup":>12} {"Efficiency":>12}')
        print(f'{"-" * 70}')

        for result in results:
            print(
                f'{result["num_workers"]:>10d} '
                f'{result["speedup"]:>11.2f}x '
                f'{result["efficiency"]:>11.1%}'
            )

        print(f'{"=" * 70}\n')

        # Efficiency should generally decrease with more workers due to overhead
        # but should still be positive
        assert all(r['efficiency'] > 0 for r in results)

    @pytest.mark.skip(reason='Benchmark test - takes too long for regular unit tests')
    @pytest.mark.slow
    def test_benchmark_full_dataset(self, temp_folder):
        """
        Benchmark processing the entire dataset.

        This test is marked as 'slow' and processes all molecules.
        Run with: pytest -m "benchmark and slow"
        """
        print(f'\n{"=" * 70}')
        print(f'Full Dataset Benchmark: {self.DATASET_NAME}')
        print(f'{"=" * 70}\n')

        worker_counts = [0, 2, 4]
        results = []

        for num_workers in worker_counts:
            print(
                f'Processing full dataset with {num_workers} workers...',
                end=' ',
                flush=True,
            )

            elapsed, count = self._benchmark_worker_count(
                num_workers=num_workers,
                temp_folder=temp_folder,
                num_molecules=None,  # Process all
            )

            throughput = count / elapsed
            results.append(
                {
                    'num_workers': num_workers,
                    'elapsed': elapsed,
                    'count': count,
                    'throughput': throughput,
                }
            )

            print(f'{elapsed:.3f}s ({count} molecules, {throughput:.1f} mol/s)')

        print(f'\n{"=" * 70}')
        print(f'Processed {results[0]["count"]} molecules total')
        print(f'{"=" * 70}\n')


# @pytest.mark.skip(reason="Benchmark test - takes too long for regular unit tests")
@pytest.mark.benchmark
def test_benchmark_with_plot():
    """
    Comprehensive benchmark that generates a visual plot of results.

    This test benchmarks different worker counts and creates a matplotlib
    figure showing timing and throughput results.

    The plot is saved to tests/artifacts/ directory.

    Run with: pytest tests/test_dataset_benchmark.py::test_benchmark_with_plot -m benchmark -v
    """
    print(f'\n{"=" * 70}')
    print('Comprehensive Benchmark with Visualization')
    print(f'{"=" * 70}\n')

    dataset_name = 'aqsoldb'
    num_molecules = 9000  # Process 200 molecules for good statistics
    worker_counts = [0, 1, 2, 4, 8]

    results = []

    # Create temporary directory for dataset storage
    with tempfile.TemporaryDirectory() as tmp_path:
        # Run benchmarks
        for num_workers in worker_counts:
            print(f'Benchmarking {num_workers} workers...', end=' ', flush=True)

            dataset = GraphDataset(
                dataset=dataset_name,
                num_workers=num_workers,
                smiles_column='SMILES',
                target_columns=['Solubility'],
                folder_path=tmp_path,
                buffer_size=100,
            )

            start_time = time.time()
            count = 0
            for i, graph in enumerate(dataset):
                count += 1
                if i + 1 >= num_molecules:
                    break
            elapsed = time.time() - start_time

            dataset.close()

            throughput = count / elapsed
            results.append(
                {
                    'num_workers': num_workers,
                    'elapsed': elapsed,
                    'throughput': throughput,
                    'speedup': results[0]['elapsed'] / elapsed if results else 1.0,
                }
            )

            print(f'{elapsed:.3f}s ({throughput:.1f} mol/s)')

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f'GraphDataset Performance Benchmark\n{dataset_name} dataset, {num_molecules} molecules',
        fontsize=14,
        fontweight='bold',
    )

    workers = [r['num_workers'] for r in results]
    times = [r['elapsed'] for r in results]
    throughputs = [r['throughput'] for r in results]
    speedups = [r['speedup'] for r in results]

    # Plot 1: Processing Time
    ax1 = axes[0, 0]
    ax1.plot(workers, times, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Number of Workers', fontsize=11)
    ax1.set_ylabel('Time (seconds)', fontsize=11)
    ax1.set_title('Processing Time vs Workers', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(workers)

    # Plot 2: Throughput
    ax2 = axes[0, 1]
    ax2.plot(workers, throughputs, 'o-', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Number of Workers', fontsize=11)
    ax2.set_ylabel('Throughput (molecules/sec)', fontsize=11)
    ax2.set_title('Throughput vs Workers', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(workers)

    # Plot 3: Speedup
    ax3 = axes[1, 0]
    ax3.plot(
        workers,
        speedups,
        'o-',
        linewidth=2,
        markersize=8,
        color='#F18F01',
        label='Actual',
    )
    # Add ideal linear speedup line
    ideal_speedup = [1 if w == 0 else w / max(1, workers[1]) for w in workers]
    ax3.plot(
        workers,
        ideal_speedup,
        '--',
        linewidth=2,
        color='gray',
        alpha=0.5,
        label='Ideal (linear)',
    )
    ax3.set_xlabel('Number of Workers', fontsize=11)
    ax3.set_ylabel('Speedup vs Sequential', fontsize=11)
    ax3.set_title('Speedup vs Workers', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(workers)
    ax3.legend()

    # Plot 4: Efficiency
    ax4 = axes[1, 1]
    # Calculate efficiency (speedup / num_workers), accounting for sequential (0 workers)
    efficiencies = []
    for r in results:
        if r['num_workers'] == 0:
            efficiencies.append(1.0)  # Sequential is 100% "efficient"
        else:
            efficiencies.append(
                r['speedup'] / r['num_workers'] if r['num_workers'] > 0 else 0
            )

    ax4.plot(
        workers,
        [e * 100 for e in efficiencies],
        'o-',
        linewidth=2,
        markersize=8,
        color='#6A994E',
    )
    ax4.axhline(
        y=100,
        color='gray',
        linestyle='--',
        linewidth=2,
        alpha=0.5,
        label='Ideal (100%)',
    )
    ax4.set_xlabel('Number of Workers', fontsize=11)
    ax4.set_ylabel('Parallel Efficiency (%)', fontsize=11)
    ax4.set_title('Parallel Efficiency vs Workers', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(workers)
    ax4.set_ylim([0, 120])
    ax4.legend()

    plt.tight_layout()

    # Ensure artifacts directory exists
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)

    # Save the plot to artifacts directory
    output_path = os.path.join(ARTIFACTS_PATH, 'benchmark_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'\n✓ Benchmark plot saved to: {output_path}')

    # Also save as PDF for publication quality
    output_pdf = os.path.join(ARTIFACTS_PATH, 'benchmark_results.pdf')
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f'✓ Benchmark plot saved to: {output_pdf}')

    # Print summary table
    print(f'\n{"=" * 70}')
    print('Benchmark Results Summary:')
    print(f'{"=" * 70}')
    print(
        f'{"Workers":>8} {"Time (s)":>10} {"Throughput":>15} {"Speedup":>10} {"Efficiency":>12}'
    )
    print(f'{"-" * 70}')
    for r, eff in zip(results, efficiencies):
        print(
            f'{r["num_workers"]:>8d} '
            f'{r["elapsed"]:>10.3f} '
            f'{r["throughput"]:>12.1f} mol/s '
            f'{r["speedup"]:>9.2f}x '
            f'{eff:>11.1%}'
        )
    print(f'{"=" * 70}\n')

    plt.close()

    # Assertions
    assert len(results) == len(worker_counts)
    assert all(r['elapsed'] > 0 for r in results)
    assert os.path.exists(output_path), f'PNG plot not found at {output_path}'
    assert os.path.exists(output_pdf), f'PDF plot not found at {output_pdf}'


@pytest.mark.skip(reason='Benchmark test - takes too long for regular unit tests')
@pytest.mark.benchmark
def test_benchmark_processing_and_queue_times(temp_folder=None):
    """
    Benchmark SMILES processing and multiprocessing queue transfer times.

    This test measures:
    1. Time to process individual SMILES strings into graph dicts
    2. Time to transfer graph dicts through multiprocessing queues (put operation)
    3. Time to retrieve graph dicts from multiprocessing queues (get operation)

    Results are visualized as violin plots showing the distribution of timings
    across multiple molecules.

    Run with: pytest tests/test_dataset_benchmark.py::test_benchmark_processing_and_queue_times -m benchmark -v
    """
    print(f'\n{"=" * 70}')
    print('Processing and Queue Transfer Benchmark')
    print(f'{"=" * 70}\n')

    dataset_name = 'aqsoldb'
    num_molecules = 2000  # Sample size for good statistics

    # Create temporary directory if not provided
    if temp_folder is None:
        temp_folder = tempfile.gettempdir()

    print(f'Loading {dataset_name} dataset...', end=' ', flush=True)

    # Load the SMILES dataset
    df = load_smiles_dataset(
        dataset_name=dataset_name,
        folder_path=temp_folder,
        use_cache=True,
    )

    # Sample molecules for benchmarking
    df_sample = df.head(num_molecules)
    smiles_list = df_sample['SMILES'].tolist()

    print(f'loaded {len(smiles_list)} molecules')

    # Initialize processing instance
    processing = MoleculeProcessing()

    # === BENCHMARK 1: Processing Times ===
    print('\nBenchmarking SMILES processing...', end=' ', flush=True)

    processing_times = []
    graphs = []

    for smiles in smiles_list:
        try:
            start = time.perf_counter()
            graph = processing.process(smiles)
            elapsed = time.perf_counter() - start

            processing_times.append(elapsed)
            graphs.append(graph)
        except Exception as e:
            # Skip invalid SMILES
            print(f'\nWarning: Failed to process SMILES: {smiles[:50]}... ({e})')
            continue

    print(f'completed ({len(processing_times)} successful)')

    # === BENCHMARK 2: Queue Transfer Times ===
    print('Benchmarking queue transfers...', end=' ', flush=True)

    queue_put_times = []
    queue_get_times = []

    # Create a multiprocessing queue
    q = mp.Queue()

    for graph in graphs:
        # Measure put time
        start = time.perf_counter()
        q.put(graph)
        put_time = time.perf_counter() - start
        queue_put_times.append(put_time)

        # Measure get time
        start = time.perf_counter()
        _ = q.get()
        get_time = time.perf_counter() - start
        queue_get_times.append(get_time)

    print('completed')

    # === STATISTICS ===
    processing_times = np.array(processing_times) * 1000  # Convert to milliseconds
    queue_put_times = np.array(queue_put_times) * 1000
    queue_get_times = np.array(queue_get_times) * 1000

    # Print summary statistics
    print(f'\n{"=" * 70}')
    print('Timing Statistics (milliseconds):')
    print(f'{"=" * 70}')
    print(
        f'{"Operation":<25} {"Mean":>10} {"Median":>10} {"Std Dev":>10} {"Min":>10} {"Max":>10}'
    )
    print(f'{"-" * 70}')

    stats = [
        ('SMILES Processing', processing_times),
        ('Queue Put', queue_put_times),
        ('Queue Get', queue_get_times),
    ]

    for name, times in stats:
        print(
            f'{name:<25} '
            f'{np.mean(times):>10.3f} '
            f'{np.median(times):>10.3f} '
            f'{np.std(times):>10.3f} '
            f'{np.min(times):>10.3f} '
            f'{np.max(times):>10.3f}'
        )

    print(f'{"=" * 70}\n')

    # === VISUALIZATION ===
    print('Creating violin plots...', end=' ', flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f'SMILES Processing and Queue Transfer Benchmark\n{dataset_name} dataset, {len(graphs)} molecules',
        fontsize=14,
        fontweight='bold',
    )

    # Plot 1: Processing times
    ax1 = axes[0]
    parts1 = ax1.violinplot(
        [processing_times], positions=[0], showmeans=True, showmedians=True
    )
    ax1.set_ylabel('Time (milliseconds)', fontsize=11)
    ax1.set_title('SMILES Processing Time', fontsize=12, fontweight='bold')
    ax1.set_xticks([])
    ax1.grid(True, alpha=0.3, axis='y')

    # Add text annotations
    ax1.text(
        0.5,
        0.95,
        f'Mean: {np.mean(processing_times):.3f} ms\nMedian: {np.median(processing_times):.3f} ms',
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )

    # Plot 2: Queue put times
    ax2 = axes[1]
    parts2 = ax2.violinplot(
        [queue_put_times], positions=[0], showmeans=True, showmedians=True
    )
    ax2.set_ylabel('Time (milliseconds)', fontsize=11)
    ax2.set_title('Queue Put Time', fontsize=12, fontweight='bold')
    ax2.set_xticks([])
    ax2.grid(True, alpha=0.3, axis='y')

    ax2.text(
        0.5,
        0.95,
        f'Mean: {np.mean(queue_put_times):.3f} ms\nMedian: {np.median(queue_put_times):.3f} ms',
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )

    # Plot 3: Queue get times
    ax3 = axes[2]
    parts3 = ax3.violinplot(
        [queue_get_times], positions=[0], showmeans=True, showmedians=True
    )
    ax3.set_ylabel('Time (milliseconds)', fontsize=11)
    ax3.set_title('Queue Get Time', fontsize=12, fontweight='bold')
    ax3.set_xticks([])
    ax3.grid(True, alpha=0.3, axis='y')

    ax3.text(
        0.5,
        0.95,
        f'Mean: {np.mean(queue_get_times):.3f} ms\nMedian: {np.median(queue_get_times):.3f} ms',
        transform=ax3.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )

    plt.tight_layout()

    # Ensure artifacts directory exists
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)

    # Save the plots
    output_path = os.path.join(ARTIFACTS_PATH, 'processing_queue_benchmark.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print('✓')
    print(f'✓ Benchmark plot saved to: {output_path}')

    # Also save as PDF for publication quality
    output_pdf = os.path.join(ARTIFACTS_PATH, 'processing_queue_benchmark.pdf')
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f'✓ Benchmark plot saved to: {output_pdf}\n')

    plt.close()

    # Cleanup
    q.close()

    # Assertions
    assert len(processing_times) > 0, 'No molecules were successfully processed'
    assert len(processing_times) == len(queue_put_times) == len(queue_get_times)
    assert np.all(processing_times > 0), 'All processing times should be positive'
    assert np.all(queue_put_times >= 0), 'All queue put times should be non-negative'
    assert np.all(queue_get_times >= 0), 'All queue get times should be non-negative'
    assert os.path.exists(output_path), f'PNG plot not found at {output_path}'
    assert os.path.exists(output_pdf), f'PDF plot not found at {output_pdf}'


if __name__ == '__main__':
    # Run with: python tests/test_dataset_benchmark.py
    pytest.main([__file__, '-m', 'benchmark', '-v', '-s'])
