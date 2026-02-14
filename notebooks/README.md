# Analysis Notebooks

This directory contains Jupyter notebooks for analyzing benchmark results from the vLLM inference server.

## Available Notebooks

### benchmark_analysis.ipynb
Comprehensive analysis notebook that includes:
- Loading benchmark results from JSON/CSV files
- Statistical analysis (TTFT, ITL, throughput)
- Visualizations with Plotly
- GPU telemetry analysis (VRAM, power, temperature)
- Context length comparison
- Multi-run comparison

## Setup

### Install Dependencies

```bash
pip install jupyter pandas numpy plotly
```

### Running Jupyter

```bash
cd notebooks
jupyter notebook
```

Then open `benchmark_analysis.ipynb` in your browser.

## Usage Workflow

1. **Run Benchmarks** on the server using docker-compose
2. **Copy Results** to your local machine:
   ```bash
   rsync -avz server:/path/to/InferenceServer/results/ ../results/
   ```
3. **Open Notebook** and run all cells
4. **Customize Analysis** by modifying cells or adding new visualizations

## Key Visualizations

The notebook generates:
- TTFT distribution histograms and box plots
- ITL scatter plots over request sequence
- Throughput comparisons across context lengths
- GPU metrics over time (utilization, VRAM, temperature, power)
- Multi-metric comparison charts
- Percentile analysis (P50, P95, P99)

## Tips

- Update the `RESULT_FILE` variable to analyze specific benchmark runs
- Use the comparison section to track performance changes over time
- Export summaries for long-term tracking
- Modify visualizations by changing Plotly parameters

## Requirements

- pandas
- numpy
- plotly
- jupyter
