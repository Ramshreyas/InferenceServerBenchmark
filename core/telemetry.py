#!/usr/bin/env python3
"""
GPU telemetry collection using nvidia-smi.
Tracks VRAM usage, power draw, temperature, and utilization.
"""

import subprocess
import json
import threading
import time
from typing import Dict, List, Optional
from datetime import datetime


class TelemetryCollector:
    """Collects GPU telemetry data during benchmark execution."""
    
    def __init__(self):
        self.data: List[Dict] = []
        self.collecting = False
        self.thread: Optional[threading.Thread] = None
        
    def _collect_gpu_stats(self) -> Dict:
        """Query nvidia-smi for current GPU statistics."""
        try:
            # Query nvidia-smi for JSON output
            cmd = [
                'nvidia-smi',
                '--query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,temperature.gpu,'
                'utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,'
                'power.draw,power.limit,clocks.current.graphics,clocks.current.memory',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse CSV output
            values = result.stdout.strip().split(', ')
            
            return {
                'timestamp': datetime.now().isoformat(),
                'gpu_name': values[1],
                'pci_bus_id': values[2],
                'driver_version': values[3],
                'power_state': values[4],
                'temperature_c': float(values[5]),
                'gpu_utilization_percent': float(values[6]),
                'memory_utilization_percent': float(values[7]),
                'memory_total_mb': float(values[8]),
                'memory_free_mb': float(values[9]),
                'memory_used_mb': float(values[10]),
                'power_draw_w': float(values[11]),
                'power_limit_w': float(values[12]),
                'graphics_clock_mhz': float(values[13]),
                'memory_clock_mhz': float(values[14])
            }
            
        except subprocess.CalledProcessError as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'error': f"nvidia-smi failed: {e}"
            }
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'error': f"Failed to parse nvidia-smi output: {e}"
            }
    
    def _collection_loop(self, interval: float):
        """Background thread that periodically collects GPU stats."""
        while self.collecting:
            stats = self._collect_gpu_stats()
            self.data.append(stats)
            time.sleep(interval)
    
    def start_collection(self, interval: float = 1.0):
        """Start collecting GPU telemetry in background thread."""
        if self.collecting:
            return
        
        self.collecting = True
        self.thread = threading.Thread(target=self._collection_loop, args=(interval,), daemon=True)
        self.thread.start()
    
    def stop_collection(self):
        """Stop telemetry collection."""
        self.collecting = False
        if self.thread:
            self.thread.join(timeout=5.0)
    
    def get_data(self) -> List[Dict]:
        """Return collected telemetry data."""
        return self.data
    
    def get_summary(self) -> Dict:
        """Calculate summary statistics from collected data."""
        if not self.data:
            return {}
        
        valid_data = [d for d in self.data if 'error' not in d]
        
        if not valid_data:
            return {'error': 'No valid telemetry data collected'}
        
        # Extract metrics
        temps = [d['temperature_c'] for d in valid_data]
        gpu_utils = [d['gpu_utilization_percent'] for d in valid_data]
        mem_used = [d['memory_used_mb'] for d in valid_data]
        power_draws = [d['power_draw_w'] for d in valid_data]
        
        return {
            'gpu_name': valid_data[0]['gpu_name'],
            'samples_collected': len(valid_data),
            'temperature': {
                'mean_c': sum(temps) / len(temps),
                'max_c': max(temps),
                'min_c': min(temps)
            },
            'gpu_utilization': {
                'mean_percent': sum(gpu_utils) / len(gpu_utils),
                'max_percent': max(gpu_utils),
                'min_percent': min(gpu_utils)
            },
            'vram_usage': {
                'mean_mb': sum(mem_used) / len(mem_used),
                'peak_mb': max(mem_used),
                'min_mb': min(mem_used)
            },
            'power_draw': {
                'mean_w': sum(power_draws) / len(power_draws),
                'peak_w': max(power_draws),
                'min_w': min(power_draws)
            }
        }


def get_gpu_info() -> Dict:
    """Get static GPU information."""
    try:
        cmd = [
            'nvidia-smi',
            '--query-gpu=name,driver_version,memory.total,compute_cap',
            '--format=csv,noheader,nounits'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        values = result.stdout.strip().split(', ')
        
        return {
            'gpu_name': values[0],
            'driver_version': values[1],
            'memory_total_mb': float(values[2]),
            'compute_capability': values[3]
        }
    except Exception as e:
        return {'error': str(e)}


if __name__ == '__main__':
    # Test telemetry collection
    print("GPU Info:", json.dumps(get_gpu_info(), indent=2))
    
    collector = TelemetryCollector()
    print("\nCollecting telemetry for 10 seconds...")
    collector.start_collection(interval=1.0)
    time.sleep(10)
    collector.stop_collection()
    
    print(f"\nCollected {len(collector.get_data())} samples")
    print("\nSummary:", json.dumps(collector.get_summary(), indent=2))
