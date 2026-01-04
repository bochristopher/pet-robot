#!/usr/bin/env python3
"""
Benchmark wrapper for custom smart_explore.py
Measures performance metrics for comparison with ROS2 Nav2.
"""

import sys
import time
import json
import math
import threading

# Add robot_pet to path
sys.path.insert(0, '/home/bo/robot_pet')

# Metrics tracking
class BenchmarkMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.cells_mapped = []  # (timestamp, count)
        self.moves = 0
        self.stuck_events = 0
        self.emergency_stops = 0
        self.distance_traveled = 0.0
        self.last_position = None
        self.running = True

    def update_position(self, x, y):
        """Track distance traveled."""
        if self.last_position is not None:
            dx = x - self.last_position[0]
            dy = y - self.last_position[1]
            self.distance_traveled += math.sqrt(dx*dx + dy*dy)
        self.last_position = (x, y)

    def log_mapping(self, cell_count):
        """Log mapping progress."""
        elapsed = time.time() - self.start_time
        self.cells_mapped.append((elapsed, cell_count))

    def save(self, filename=None):
        """Save benchmark results."""
        if filename is None:
            filename = f'/home/bo/robot_pet/benchmark_custom_{int(time.time())}.json'

        elapsed = time.time() - self.start_time
        final_cells = self.cells_mapped[-1][1] if self.cells_mapped else 0

        results = {
            'system': 'Custom_Smart_Explorer',
            'total_time': elapsed,
            'final_cells_mapped': final_cells,
            'moves': self.moves,
            'stuck_events': self.stuck_events,
            'emergency_stops': self.emergency_stops,
            'distance_traveled': self.distance_traveled,
            'mapping_history': self.cells_mapped,
            'efficiency': final_cells / max(self.distance_traveled, 0.1),
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        print(f'\n=== BENCHMARK RESULTS ===')
        print(f'Time: {elapsed:.1f}s')
        print(f'Cells Mapped: {final_cells}')
        print(f'Moves: {self.moves}')
        print(f'Stuck Events: {self.stuck_events}')
        print(f'Emergency Stops: {self.emergency_stops}')
        print(f'Distance: {self.distance_traveled:.2f}m')
        print(f'Efficiency: {results["efficiency"]:.1f} cells/m')
        print(f'Saved to: {filename}')

        return results

# Global metrics instance
metrics = BenchmarkMetrics()

def run_benchmark(duration=300):
    """
    Run smart_explore.py with benchmarking for specified duration.

    Usage:
        python3 benchmark_explorer.py [duration_seconds]
    """
    import subprocess
    import signal

    print(f'Starting benchmark for {duration} seconds...')
    print('Press Ctrl+C to stop early\n')

    # Start smart_explore in subprocess
    proc = subprocess.Popen(
        ['python3', '/home/bo/robot_pet/smart_explore.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    start = time.time()

    try:
        while time.time() - start < duration:
            line = proc.stdout.readline()
            if not line:
                break

            print(line, end='')

            # Parse metrics from output
            if 'Move' in line and 'Mapped:' in line:
                metrics.moves += 1

                # Extract mapped cells
                try:
                    mapped_idx = line.find('Mapped:')
                    if mapped_idx > 0:
                        rest = line[mapped_idx+7:].strip()
                        cells = int(rest.split()[0])
                        metrics.log_mapping(cells)
                except:
                    pass

                # Extract position
                try:
                    pos_idx = line.find('Pos:(')
                    if pos_idx > 0:
                        pos_str = line[pos_idx+5:line.find(')', pos_idx)]
                        x, y = map(float, pos_str.split(','))
                        metrics.update_position(x, y)
                except:
                    pass

            if 'STUCK' in line:
                metrics.stuck_events += 1

            if 'EMERGENCY STOP' in line:
                metrics.emergency_stops += 1

    except KeyboardInterrupt:
        print('\nStopping benchmark...')
    finally:
        proc.terminate()
        proc.wait()
        metrics.save()


if __name__ == '__main__':
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    run_benchmark(duration)
