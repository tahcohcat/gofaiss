#!/usr/bin/env python3
"""
Aggregate multiple benchmark runs and compute statistics
Produces mean, std dev, and confidence intervals
"""

import json
import sys
import math
from collections import defaultdict
from pathlib import Path

def load_json(filename):
    """Load benchmark results from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def mean(values):
    """Calculate mean"""
    return sum(values) / len(values) if values else 0

def std_dev(values):
    """Calculate standard deviation"""
    if len(values) < 2:
        return 0
    avg = mean(values)
    variance = sum((x - avg) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)

def confidence_interval_95(values):
    """Calculate 95% confidence interval"""
    if len(values) < 2:
        return 0
    # t-value for 95% CI with n-1 degrees of freedom (approximation)
    t_values = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 
                7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
    n = len(values)
    t = t_values.get(n, 2.0)  # Use 2.0 for larger samples
    
    std = std_dev(values)
    margin = t * std / math.sqrt(n)
    return margin

def aggregate_results(files):
    """Aggregate results from multiple benchmark runs"""
    
    # Group results by (Library, IndexType)
    grouped = defaultdict(lambda: defaultdict(list))
    
    for filename in files:
        results = load_json(filename)
        for result in results:
            key = (result['Library'], result['IndexType'])
            grouped[key]['BuildTimeMs'].append(result['BuildTimeMs'])
            grouped[key]['AvgQueryMs'].append(result['AvgQueryMs'])
            grouped[key]['P50QueryMs'].append(result['P50QueryMs'])
            grouped[key]['P95QueryMs'].append(result['P95QueryMs'])
            grouped[key]['P99QueryMs'].append(result['P99QueryMs'])
            grouped[key]['QPS'].append(result['QPS'])
            grouped[key]['MemoryMB'].append(result['MemoryMB'])
            grouped[key]['Recall'].append(result['Recall'])
            
            # Store metadata (same for all runs)
            if 'metadata' not in grouped[key]:
                grouped[key]['metadata'] = {
                    'NumVectors': result['NumVectors'],
                    'Dimension': result['Dimension'],
                    'IndexParams': result.get('IndexParams', {}),
                }
    
    # Compute statistics for each group
    aggregated = []
    
    for (library, index_type), metrics in grouped.items():
        result = {
            'Library': library,
            'IndexType': index_type,
            'NumRuns': len(metrics['QPS']),
            'NumVectors': metrics['metadata']['NumVectors'],
            'Dimension': metrics['metadata']['Dimension'],
            'IndexParams': metrics['metadata']['IndexParams'],
            
            # Mean values
            'BuildTimeMs': {
                'mean': mean(metrics['BuildTimeMs']),
                'std': std_dev(metrics['BuildTimeMs']),
                'ci95': confidence_interval_95(metrics['BuildTimeMs']),
                'min': min(metrics['BuildTimeMs']),
                'max': max(metrics['BuildTimeMs']),
            },
            'AvgQueryMs': {
                'mean': mean(metrics['AvgQueryMs']),
                'std': std_dev(metrics['AvgQueryMs']),
                'ci95': confidence_interval_95(metrics['AvgQueryMs']),
                'min': min(metrics['AvgQueryMs']),
                'max': max(metrics['AvgQueryMs']),
            },
            'P50QueryMs': {
                'mean': mean(metrics['P50QueryMs']),
                'std': std_dev(metrics['P50QueryMs']),
                'ci95': confidence_interval_95(metrics['P50QueryMs']),
            },
            'P95QueryMs': {
                'mean': mean(metrics['P95QueryMs']),
                'std': std_dev(metrics['P95QueryMs']),
                'ci95': confidence_interval_95(metrics['P95QueryMs']),
            },
            'P99QueryMs': {
                'mean': mean(metrics['P99QueryMs']),
                'std': std_dev(metrics['P99QueryMs']),
                'ci95': confidence_interval_95(metrics['P99QueryMs']),
            },
            'QPS': {
                'mean': mean(metrics['QPS']),
                'std': std_dev(metrics['QPS']),
                'ci95': confidence_interval_95(metrics['QPS']),
                'min': min(metrics['QPS']),
                'max': max(metrics['QPS']),
                'cv': std_dev(metrics['QPS']) / mean(metrics['QPS']) if mean(metrics['QPS']) > 0 else 0,  # Coefficient of variation
            },
            'MemoryMB': {
                'mean': mean(metrics['MemoryMB']),
                'std': std_dev(metrics['MemoryMB']),
                'ci95': confidence_interval_95(metrics['MemoryMB']),
            },
            'Recall': {
                'mean': mean(metrics['Recall']),
                'std': std_dev(metrics['Recall']),
                'ci95': confidence_interval_95(metrics['Recall']),
                'min': min(metrics['Recall']),
                'max': max(metrics['Recall']),
            },
        }
        
        aggregated.append(result)
    
    return aggregated

def print_summary(aggregated):
    """Print human-readable summary"""
    print("\n" + "="*120)
    print(f"{'Library':<15} {'Index':<10} {'QPS (mean±std)':<25} {'Recall (mean±std)':<25} {'CV':<10} {'Runs':<5}")
    print("="*120)
    
    for result in aggregated:
        qps = result['QPS']
        recall = result['Recall']
        
        print(f"{result['Library']:<15} "
              f"{result['IndexType']:<10} "
              f"{qps['mean']:>8.0f} ± {qps['std']:>6.0f} ({qps['ci95']:>6.0f}) "
              f"{recall['mean']:>6.4f} ± {recall['std']:>6.4f} ({recall['ci95']:>6.4f}) "
              f"{qps['cv']:>8.4f}  "
              f"{result['NumRuns']:<5}")
    
    print("="*120)
    print("\nNote: Values shown as mean ± std (95% CI)")
    print("CV = Coefficient of Variation (lower is more consistent)")
    print()

def print_detailed_report(aggregated, output_file=None):
    """Generate detailed markdown report"""
    
    lines = []
    lines.append("# Aggregated Benchmark Results\n")
    lines.append(f"**Number of runs:** {aggregated[0]['NumRuns']}\n")
    lines.append(f"**Configuration:** {aggregated[0]['NumVectors']:,} vectors, {aggregated[0]['Dimension']} dimensions\n\n")
    
    lines.append("## Summary Table\n\n")
    lines.append("| Library | Index | QPS (mean±CI) | Query ms (mean±CI) | Recall (mean±CI) | Memory MB |\n")
    lines.append("|---------|-------|---------------|-------------------|------------------|----------|\n")
    
    for result in aggregated:
        qps = result['QPS']
        query = result['AvgQueryMs']
        recall = result['Recall']
        memory = result['MemoryMB']
        
        lines.append(f"| {result['Library']} | {result['IndexType']} | "
                    f"{qps['mean']:.0f}±{qps['ci95']:.0f} | "
                    f"{query['mean']:.4f}±{query['ci95']:.4f} | "
                    f"{recall['mean']:.4f}±{recall['ci95']:.4f} | "
                    f"{memory['mean']:.2f} |\n")
    
    lines.append("\n## Detailed Metrics\n\n")
    
    for result in aggregated:
        lines.append(f"### {result['Library']} - {result['IndexType']}\n\n")
        
        lines.append("**Query Performance:**\n")
        qps = result['QPS']
        lines.append(f"- QPS: {qps['mean']:.0f} ± {qps['std']:.0f} (95% CI: ±{qps['ci95']:.0f})\n")
        lines.append(f"- Range: {qps['min']:.0f} - {qps['max']:.0f}\n")
        lines.append(f"- CV: {qps['cv']:.4f} ({qps['cv']*100:.2f}%)\n\n")
        
        lines.append("**Latency:**\n")
        for percentile in ['AvgQueryMs', 'P50QueryMs', 'P95QueryMs', 'P99QueryMs']:
            metric = result[percentile]
            name = percentile.replace('QueryMs', '')
            lines.append(f"- {name}: {metric['mean']:.4f} ± {metric['std']:.4f} ms\n")
        lines.append("\n")
        
        lines.append("**Build Time:**\n")
        build = result['BuildTimeMs']
        lines.append(f"- Mean: {build['mean']:.0f} ms ({build['mean']/1000:.2f}s)\n")
        lines.append(f"- Range: {build['min']:.0f} - {build['max']:.0f} ms\n\n")
        
        lines.append("**Recall:**\n")
        recall = result['Recall']
        lines.append(f"- Mean: {recall['mean']:.4f} ± {recall['std']:.4f}\n")
        lines.append(f"- Range: {recall['min']:.4f} - {recall['max']:.4f}\n\n")
        
        lines.append("**Memory:**\n")
        memory = result['MemoryMB']
        lines.append(f"- Mean: {memory['mean']:.2f} MB\n\n")
        
        if result['IndexParams']:
            lines.append("**Index Parameters:**\n")
            for k, v in result['IndexParams'].items():
                lines.append(f"- {k}: {v}\n")
            lines.append("\n")
    
    # Statistical validity check
    lines.append("\n## Statistical Validity\n\n")
    lines.append("**Coefficient of Variation (CV) Analysis:**\n")
    lines.append("- CV < 0.05: Excellent consistency\n")
    lines.append("- CV < 0.10: Good consistency\n")
    lines.append("- CV > 0.10: High variability (consider more runs)\n\n")
    
    for result in aggregated:
        cv = result['QPS']['cv']
        status = "✓ Excellent" if cv < 0.05 else "✓ Good" if cv < 0.10 else "⚠ High variability"
        lines.append(f"- {result['Library']} {result['IndexType']}: CV={cv:.4f} ({status})\n")
    
    report = ''.join(lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Detailed report saved to: {output_file}")
    
    return report

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 aggregate_runs.py <result1.json> <result2.json> ... > aggregated.json")
        print("\nExample:")
        print("  python3 aggregate_runs.py results/run_*.json > aggregated.json")
        sys.exit(1)
    
    files = sys.argv[1:]
    
    print(f"Aggregating {len(files)} benchmark runs...", file=sys.stderr)
    
    aggregated = aggregate_results(files)
    
    # Print to stderr for human reading
    print_summary(aggregated)
    
    # Output directory for detailed report
    output_dir = Path(files[0]).parent
    report_file = output_dir / "aggregated_report.md"
    print_detailed_report(aggregated, report_file)
    
    # Output JSON to stdout for piping
    print(json.dumps(aggregated, indent=2))

if __name__ == "__main__":
    main()