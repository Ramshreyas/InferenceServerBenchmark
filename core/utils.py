#!/usr/bin/env python3
"""
Utility functions for logging, formatting, and data export.
"""

import json
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger with console output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger


def format_latency(ms: float) -> str:
    """Format latency in milliseconds with appropriate precision."""
    if ms < 1:
        return f"{ms*1000:.2f} μs"
    elif ms < 1000:
        return f"{ms:.2f} ms"
    else:
        return f"{ms/1000:.2f} s"


def format_throughput(tokens_per_sec: float) -> str:
    """Format throughput in tokens per second."""
    if tokens_per_sec < 1000:
        return f"{tokens_per_sec:.2f} tokens/s"
    else:
        return f"{tokens_per_sec/1000:.2f}k tokens/s"


def format_memory(mb: float) -> str:
    """Format memory size in human-readable format."""
    if mb < 1024:
        return f"{mb:.2f} MB"
    elif mb < 1024 * 1024:
        return f"{mb/1024:.2f} GB"
    else:
        return f"{mb/(1024*1024):.2f} TB"


def save_json(data: Any, filepath: Path):
    """Save data structure as JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def save_csv(data: List[Dict], filepath: Path):
    """Save list of dictionaries as CSV file."""
    if not data:
        return
    
    # Get all unique keys from all dicts
    fieldnames = set()
    for item in data:
        fieldnames.update(item.keys())
    fieldnames = sorted(fieldnames)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def load_json(filepath: Path) -> Any:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_csv(filepath: Path) -> List[Dict]:
    """Load CSV file as list of dictionaries."""
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def calculate_percentiles(values: List[float], percentiles: List[int] = [50, 95, 99]) -> Dict[str, float]:
    """Calculate percentiles from a list of values."""
    import numpy as np
    
    if not values:
        return {}
    
    return {
        f"p{p}": np.percentile(values, p)
        for p in percentiles
    }


def aggregate_metrics(results: List[Dict], metric_key: str) -> Dict[str, float]:
    """Calculate summary statistics for a metric across results."""
    import numpy as np
    
    values = [r[metric_key] for r in results if metric_key in r and r[metric_key] is not None]
    
    if not values:
        return {}
    
    return {
        'count': len(values),
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'median': np.median(values),
        'p95': np.percentile(values, 95),
        'p99': np.percentile(values, 99)
    }


def create_markdown_report(results: List[Dict], output_path: Path):
    """Generate a markdown report from benchmark results."""
    import numpy as np
    from datetime import datetime
    
    successful = [r for r in results if r.get('success')]
    
    report = f"""# Benchmark Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total Requests: {len(results)}
- Successful: {len(successful)}
- Failed: {len(results) - len(successful)}
- Success Rate: {len(successful)/len(results)*100:.2f}%

"""
    
    if successful:
        # TTFT stats
        ttfts = [r['ttft_ms'] for r in successful if r.get('ttft_ms')]
        if ttfts:
            report += f"""## Time to First Token (TTFT)
- Mean: {format_latency(np.mean(ttfts))}
- Median: {format_latency(np.median(ttfts))}
- P95: {format_latency(np.percentile(ttfts, 95))}
- P99: {format_latency(np.percentile(ttfts, 99))}
- Min: {format_latency(np.min(ttfts))}
- Max: {format_latency(np.max(ttfts))}

"""
        
        # ITL stats
        itls = [r['itl_ms'] for r in successful if r.get('itl_ms')]
        if itls:
            report += f"""## Inter-Token Latency (ITL)
- Mean: {format_latency(np.mean(itls))}
- Median: {format_latency(np.median(itls))}
- P95: {format_latency(np.percentile(itls, 95))}
- Min: {format_latency(np.min(itls))}
- Max: {format_latency(np.max(itls))}

"""
        
        # Throughput stats
        throughputs = [r['throughput_tokens_per_sec'] for r in successful]
        if throughputs:
            report += f"""## Throughput
- Mean: {format_throughput(np.mean(throughputs))}
- Median: {format_throughput(np.median(throughputs))}
- Total Tokens Generated: {sum(r['tokens_generated'] for r in successful)}

"""
    
    with open(output_path, 'w') as f:
        f.write(report)


if __name__ == '__main__':
    # Test utilities
    logger = setup_logger('test')
    logger.info("Logger test")
    
    print(f"Latency: {format_latency(125.5)}")
    print(f"Throughput: {format_throughput(1250.75)}")
    print(f"Memory: {format_memory(49152)}")
