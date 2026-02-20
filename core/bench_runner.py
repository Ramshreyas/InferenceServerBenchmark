#!/usr/bin/env python3
"""
Main benchmarking runner for vLLM inference server.
Loads config, generates requests, collects metrics, and exports results.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import yaml
import numpy as np
from openai import OpenAI
from tqdm import tqdm

from utils import setup_logger, format_latency, format_throughput, save_json, save_csv
from telemetry import TelemetryCollector


class BenchmarkRunner:
    """Orchestrates benchmark execution and result collection."""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = setup_logger(self.config['benchmark']['name'])
        self.output_dir = Path('/results')
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize OpenAI client (vLLM compatible)
        vllm_endpoint = os.getenv('VLLM_ENDPOINT', 'http://localhost:8000/v1')
        self.client = OpenAI(
            base_url=vllm_endpoint,
            api_key="dummy"  # vLLM doesn't require real API key
        )
        
        self.telemetry = TelemetryCollector() if self.config.get('telemetry', {}).get('collect_gpu_stats') else None
        
        # Resolve model name: use config value, or auto-detect from vLLM if 'auto'
        configured_name = self.config['model']['name']
        if configured_name == 'auto':
            self._model_name = self._detect_model_name()
        else:
            self._model_name = configured_name
        
    def _detect_model_name(self) -> str:
        """Query vLLM for the name of the currently loaded model."""
        try:
            models = self.client.models.list()
            name = models.data[0].id
            self.logger.info(f"Auto-detected model: {name}")
            return name
        except Exception as e:
            raise RuntimeError(f"Could not auto-detect model name from vLLM: {e}")

    def _load_config(self, config_path: str) -> Dict:
        """Load YAML configuration file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _generate_prompt(self, num_tokens: int) -> str:
        """Generate a synthetic prompt of approximately num_tokens length."""
        # Approximate: 1 token ~= 4 characters
        base_text = "Explain the following concept in detail with examples and use cases: "
        filler = "artificial intelligence and machine learning in modern applications " * (num_tokens // 10)
        return base_text + filler[:num_tokens * 4]
    
    def _run_single_request(self, prompt: str, max_tokens: int, request_id: int) -> Dict[str, Any]:
        """Execute a single inference request and measure metrics."""
        start_time = time.perf_counter()
        
        try:
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": self.config['requests'].get('system_prompt', '')},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                stream=True
            )
            
            # Collect streaming tokens
            first_token_time = None
            token_times = []
            tokens = []
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    token_time = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = token_time
                    token_times.append(token_time)
                    tokens.append(chunk.choices[0].delta.content)
            
            end_time = time.perf_counter()
            
            # Calculate metrics
            ttft = (first_token_time - start_time) * 1000 if first_token_time else None  # ms
            total_latency = (end_time - start_time) * 1000  # ms
            num_tokens_generated = len(tokens)
            
            # Inter-token latency (average time between consecutive tokens)
            if len(token_times) > 1:
                itl = np.mean(np.diff(token_times)) * 1000  # ms
            else:
                itl = None
            
            throughput = num_tokens_generated / (end_time - start_time) if (end_time - start_time) > 0 else 0
            
            return {
                'request_id': request_id,
                'success': True,
                'ttft_ms': ttft,
                'itl_ms': itl,
                'total_latency_ms': total_latency,
                'tokens_generated': num_tokens_generated,
                'throughput_tokens_per_sec': throughput,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Request {request_id} failed: {e}")
            return {
                'request_id': request_id,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _run_benchmark_sweep(self, context_length: int) -> List[Dict]:
        """Run benchmark for a specific context length."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Running benchmark for context length: {context_length}")
        self.logger.info(f"{'='*60}")
        
        config = self.config['requests']
        num_requests = config['num_requests']
        
        results = []
        
        # Start telemetry collection
        if self.telemetry:
            self.telemetry.start_collection(interval=self.config['telemetry']['sample_interval_sec'])
        
        # Generate and execute requests
        with tqdm(total=num_requests, desc=f"Context {context_length}") as pbar:
            for i in range(num_requests):
                # Generate prompt
                prompt_tokens = np.random.randint(
                    config['prompt_tokens_min'],
                    config['prompt_tokens_max']
                )
                prompt = self._generate_prompt(prompt_tokens)
                
                # Execute request
                result = self._run_single_request(
                    prompt=prompt,
                    max_tokens=config['completion_tokens'],
                    request_id=i
                )
                result['context_length'] = context_length
                result['prompt_tokens'] = prompt_tokens
                results.append(result)
                
                pbar.update(1)
                
                # Inter-request delay for rate limiting
                if config.get('arrival_pattern') == 'poisson' and config.get('rate_per_second'):
                    delay = np.random.exponential(1.0 / config['rate_per_second'])
                    time.sleep(delay)
        
        # Stop telemetry collection
        if self.telemetry:
            self.telemetry.stop_collection()
        
        return results
    
    def run(self):
        """Main execution entry point."""
        self.logger.info(f"Starting benchmark: {self.config['benchmark']['name']}")
        self.logger.info(f"Description: {self.config['benchmark']['description']}")
        
        # Wait for vLLM server to be ready
        self.logger.info("Waiting for vLLM server...")
        self._wait_for_server()
        
        all_results = []
        
        # Run benchmark for each context length
        context_lengths = self.config.get('context_lengths', [8192])
        for ctx_len in context_lengths:
            results = self._run_benchmark_sweep(ctx_len)
            all_results.extend(results)
        
        # Aggregate and save results
        self._save_results(all_results)
        self._print_summary(all_results)
        
        self.logger.info(f"\n✓ Benchmark complete. Results saved to {self.output_dir}")
    
    def _wait_for_server(self, timeout: int = 300):
        """Wait for vLLM server to become ready."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                # Try a simple request
                self.client.models.list()
                self.logger.info("✓ vLLM server is ready")
                return
            except Exception:
                time.sleep(5)
        raise TimeoutError("vLLM server did not become ready in time")
    
    def _save_results(self, results: List[Dict]):
        """Save benchmark results to JSON and CSV."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = self.config['benchmark']['output_prefix']
        
        # Save detailed results as JSON
        json_path = self.output_dir / f"{prefix}_{timestamp}_detailed.json"
        save_json(results, json_path)
        
        # Save summary as CSV
        csv_path = self.output_dir / f"{prefix}_{timestamp}_summary.csv"
        save_csv(results, csv_path)
        
        # Save telemetry if collected
        if self.telemetry:
            telemetry_data = self.telemetry.get_data()
            telemetry_path = self.output_dir / f"{prefix}_{timestamp}_telemetry.json"
            save_json(telemetry_data, telemetry_path)
        
        self.logger.info(f"Results saved to {json_path}")
        self.logger.info(f"Summary saved to {csv_path}")
    
    def _print_summary(self, results: List[Dict]):
        """Print aggregated benchmark summary."""
        successful = [r for r in results if r.get('success')]
        
        if not successful:
            self.logger.error("No successful requests!")
            return
        
        # Calculate statistics
        ttfts = [r['ttft_ms'] for r in successful if r.get('ttft_ms')]
        itls = [r['itl_ms'] for r in successful if r.get('itl_ms')]
        throughputs = [r['throughput_tokens_per_sec'] for r in successful]
        
        self.logger.info("\n" + "="*60)
        self.logger.info("BENCHMARK SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Total Requests: {len(results)}")
        self.logger.info(f"Successful: {len(successful)}")
        self.logger.info(f"Failed: {len(results) - len(successful)}")
        
        if ttfts:
            self.logger.info(f"\nTime to First Token (TTFT):")
            self.logger.info(f"  Mean: {format_latency(np.mean(ttfts))}")
            self.logger.info(f"  P50:  {format_latency(np.percentile(ttfts, 50))}")
            self.logger.info(f"  P95:  {format_latency(np.percentile(ttfts, 95))}")
            self.logger.info(f"  P99:  {format_latency(np.percentile(ttfts, 99))}")
        
        if itls:
            self.logger.info(f"\nInter-Token Latency (ITL):")
            self.logger.info(f"  Mean: {format_latency(np.mean(itls))}")
            self.logger.info(f"  P50:  {format_latency(np.percentile(itls, 50))}")
            self.logger.info(f"  P95:  {format_latency(np.percentile(itls, 95))}")
        
        if throughputs:
            self.logger.info(f"\nThroughput:")
            self.logger.info(f"  Mean: {format_throughput(np.mean(throughputs))}")
            self.logger.info(f"  Total tokens: {sum(r['tokens_generated'] for r in successful)}")
        
        self.logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description='Run vLLM benchmark')
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    args = parser.parse_args()
    
    runner = BenchmarkRunner(args.config)
    runner.run()


if __name__ == '__main__':
    main()
