#!/usr/bin/env python3
"""
Visualization script for GoFAISS benchmark results
Generates comparison charts and performance analysis
"""

import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(filename):
    """Load benchmark results from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def plot_qps_comparison(results, output_dir):
    """Plot QPS comparison across all indices"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group by library
    gofaiss = [r for r in results if r['Library'] == 'GoFAISS']
    hnswlib = [r for r in results if r['Library'] == 'hnswlib-go']
    
    x = np.arange(len(gofaiss))
    width = 0.35
    
    gofaiss_qps = [r['QPS'] for r in gofaiss]
    gofaiss_labels = [r['IndexType'] for r in gofaiss]
    
    bars1 = ax.bar(x - width/2, gofaiss_qps, width, label='GoFAISS', color='#4CAF50')
    
    if hnswlib:
        hnswlib_qps = [r['QPS'] for r in hnswlib]
        bars2 = ax.bar(x[0] + width/2, hnswlib_qps[0], width, label='hnswlib-go', color='#2196F3')
    
    ax.set_ylabel('QPS (Queries Per Second)', fontsize=12)
    ax.set_title('Query Throughput Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(gofaiss_labels, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/qps_comparison.png', dpi=300)
    print(f"Saved: {output_dir}/qps_comparison.png")
    plt.close()

def plot_recall_vs_qps(results, output_dir):
    """Plot recall vs QPS tradeoff"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {
        'GoFAISS': {'Flat': '#4CAF50', 'HNSW': '#8BC34A', 'IVF': '#CDDC39', 
                    'PQ': '#FFEB3B', 'IVFPQ': '#FFC107'},
        'hnswlib-go': {'HNSW': '#2196F3'}
    }
    
    for result in results:
        library = result['Library']
        index_type = result['IndexType']
        color = colors.get(library, {}).get(index_type, '#999999')
        
        marker = 'o' if library == 'GoFAISS' else 's'
        size = 200 if result['QPS'] > 1000 else 100
        
        ax.scatter(result['Recall'], result['QPS'], 
                  s=size, alpha=0.7, color=color,
                  marker=marker, edgecolors='black', linewidth=1.5,
                  label=f"{library} {index_type}")
        
        # Add labels
        ax.annotate(f"{library}\n{index_type}", 
                   (result['Recall'], result['QPS']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=8, alpha=0.8)
    
    ax.set_xlabel('Recall@10', fontsize=12)
    ax.set_ylabel('QPS (log scale)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Recall vs Query Performance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # Add ideal region
    ax.axhline(y=5000, color='green', linestyle='--', alpha=0.3, label='High Performance')
    ax.axvline(x=0.95, color='orange', linestyle='--', alpha=0.3, label='High Recall')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/recall_vs_qps.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/recall_vs_qps.png")
    plt.close()

def plot_memory_usage(results, output_dir):
    """Plot memory usage comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    gofaiss = [r for r in results if r['Library'] == 'GoFAISS']
    
    x = np.arange(len(gofaiss))
    memory = [r['MemoryMB'] for r in gofaiss]
    labels = [r['IndexType'] for r in gofaiss]
    
    bars = ax.bar(x, memory, color=['#4CAF50', '#8BC34A', '#CDDC39', '#FFEB3B', '#FFC107'])
    
    ax.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax.set_title('Memory Footprint by Index Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} MB',
                ha='center', va='bottom', fontsize=9)
    
    # Add compression ratio annotations for PQ variants
    for i, result in enumerate(gofaiss):
        if 'PQ' in result['IndexType']:
            extra_info = result.get('IndexParams', {})
            if 'compressionRatio' in str(extra_info):
                ax.text(i, memory[i]/2, 
                       f"~{extra_info.get('compressionRatio', 'N/A'):.1f}x\ncompression",
                       ha='center', va='center', fontsize=8, 
                       color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/memory_usage.png', dpi=300)
    print(f"Saved: {output_dir}/memory_usage.png")
    plt.close()

def plot_latency_percentiles(results, output_dir):
    """Plot latency percentiles comparison"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Focus on approximate indices (HNSW)
    hnsw_results = [r for r in results if 'HNSW' in r['IndexType']]
    
    x = np.arange(len(hnsw_results))
    width = 0.2
    
    percentiles = ['AvgQueryMs', 'P50QueryMs', 'P95QueryMs', 'P99QueryMs']
    colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF5722']
    labels = ['Average', 'P50', 'P95', 'P99']
    
    for i, (percentile, color, label) in enumerate(zip(percentiles, colors, labels)):
        values = [r[percentile] for r in hnsw_results]
        offset = width * (i - 1.5)
        ax.bar(x + offset, values, width, label=label, color=color)
    
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Query Latency Percentiles - HNSW Indices', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['Library']}\n{r['IndexType']}" for r in hnsw_results])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/latency_percentiles.png', dpi=300)
    print(f"Saved: {output_dir}/latency_percentiles.png")
    plt.close()

def plot_build_time(results, output_dir):
    """Plot index build time comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    gofaiss = [r for r in results if r['Library'] == 'GoFAISS']
    
    x = np.arange(len(gofaiss))
    build_times = [r['BuildTimeMs'] / 1000 for r in gofaiss]  # Convert to seconds
    labels = [r['IndexType'] for r in gofaiss]
    
    bars = ax.bar(x, build_times, color=['#4CAF50', '#8BC34A', '#CDDC39', '#FFEB3B', '#FFC107'])
    
    ax.set_ylabel('Build Time (seconds)', fontsize=12)
    ax.set_title('Index Construction Time', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/build_time.png', dpi=300)
    print(f"Saved: {output_dir}/build_time.png")
    plt.close()

def generate_comparison_table(results, output_dir):
    """Generate markdown comparison table"""
    
    table = ["# GoFAISS Benchmark Results\n"]
    table.append(f"Generated: {results[0]['Timestamp']}\n")
    table.append(f"Configuration: {results[0]['NumVectors']:,} vectors, {results[0]['Dimension']} dimensions\n\n")
    
    table.append("## Performance Comparison\n\n")
    table.append("| Library | Index | Build(s) | Avg(ms) | P95(ms) | P99(ms) | QPS | Memory(MB) | Recall@10 |\n")
    table.append("|---------|-------|----------|---------|---------|---------|-----|------------|----------|\n")
    
    for r in results:
        table.append(f"| {r['Library']} | {r['IndexType']} | "
                    f"{r['BuildTimeMs']/1000:.2f} | "
                    f"{r['AvgQueryMs']:.4f} | "
                    f"{r['P95QueryMs']:.4f} | "
                    f"{r['P99QueryMs']:.4f} | "
                    f"{r['QPS']:.0f} | "
                    f"{r['MemoryMB']:.2f} | "
                    f"{r['Recall']:.4f} |\n")
    
    table.append("\n## Analysis\n\n")
    
    # Find best performers
    best_qps = max(results, key=lambda x: x['QPS'])
    best_recall = max(results, key=lambda x: x['Recall'])
    lowest_mem = min(results, key=lambda x: x['MemoryMB'])
    fastest_build = min(results, key=lambda x: x['BuildTimeMs'])
    
    table.append(f"- **Fastest Query**: {best_qps['Library']} {best_qps['IndexType']} ({best_qps['QPS']:.0f} QPS)\n")
    table.append(f"- **Best Recall**: {best_recall['Library']} {best_recall['IndexType']} ({best_recall['Recall']:.4f})\n")
    table.append(f"- **Lowest Memory**: {lowest_mem['Library']} {lowest_mem['IndexType']} ({lowest_mem['MemoryMB']:.2f} MB)\n")
    table.append(f"- **Fastest Build**: {fastest_build['Library']} {fastest_build['IndexType']} ({fastest_build['BuildTimeMs']/1000:.2f}s)\n")
    
    # GoFAISS vs hnswlib comparison
    gofaiss_hnsw = next((r for r in results if r['Library'] == 'GoFAISS' and r['IndexType'] == 'HNSW'), None)
    hnswlib_hnsw = next((r for r in results if r['Library'] == 'hnswlib-go' and r['IndexType'] == 'HNSW'), None)
    
    if gofaiss_hnsw and hnswlib_hnsw:
        qps_ratio = gofaiss_hnsw['QPS'] / hnswlib_hnsw['QPS']
        table.append(f"\n### GoFAISS vs hnswlib-go (HNSW)\n\n")
        table.append(f"- Query Speed: **{qps_ratio:.2f}x** ")
        table.append(f"({gofaiss_hnsw['QPS']:.0f} vs {hnswlib_hnsw['QPS']:.0f} QPS)\n")
        table.append(f"- Build Time: {gofaiss_hnsw['BuildTimeMs']/hnswlib_hnsw['BuildTimeMs']:.2f}x ")
        table.append(f"({gofaiss_hnsw['BuildTimeMs']/1000:.2f}s vs {hnswlib_hnsw['BuildTimeMs']/1000:.2f}s)\n")
        table.append(f"- Recall: {gofaiss_hnsw['Recall']:.4f} vs {hnswlib_hnsw['Recall']:.4f}\n")
    
    with open(f'{output_dir}/results_table.md', 'w') as f:
        f.writelines(table)
    
    print(f"Saved: {output_dir}/results_table.md")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 visualize_benchmark.py <benchmark_results.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    # Create output directory
    output_dir = Path(results_file).stem + "_charts"
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"Loading results from {results_file}...")
    results = load_results(results_file)
    
    print(f"Generating visualizations in {output_dir}/...")
    
    # Generate all plots
    plot_qps_comparison(results, output_dir)
    plot_recall_vs_qps(results, output_dir)
    plot_memory_usage(results, output_dir)
    plot_latency_percentiles(results, output_dir)
    plot_build_time(results, output_dir)
    generate_comparison_table(results, output_dir)
    
    print("\nVisualization complete!")
    print(f"Open {output_dir}/results_table.md for detailed comparison")

if __name__ == "__main__":
    main()