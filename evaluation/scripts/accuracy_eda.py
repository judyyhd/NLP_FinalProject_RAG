#!/usr/bin/env python3
"""
Exploratory Data Analysis: Pairwise Accuracy Across RAG Architectures

This script generates pairwise scatter plots comparing accuracy scores
across all 4 RAG architectures (No RAG, Vanilla RAG, InstructRAG, Self-RAG).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import combinations
import sys
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

# Add path for nlp_rag_metrics
sys.path.append('.')
from nlp_rag_metrics import exact_presence

# NYU color scheme
try:
    from nyu_colors import COLORS
except ImportError:
    COLORS = ['#57068c', '#8900e1', '#330662', '#ff6900']

# Configuration
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAMES = {
    'no_rag': 'No RAG',
    'vanilla_rag': 'Vanilla RAG',
    'instructrag': 'InstructRAG',
    'selfrag': 'Self-RAG'
}

MODEL_FILES = {
    'no_rag': 'no_rag_devfull_results.json',
    'vanilla_rag': 'vanilla_rag_devfull_results.json',
    'instructrag': 'instructrag_llama2_devfull_results.json',
}


def load_model_data(model_key):
    """Load results for a specific model."""
    file_path = RESULTS_DIR / MODEL_FILES[model_key]
    
    if not file_path.exists():
        print(f"⚠ File not found: {file_path}")
        return None
    
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    return data


def compute_accuracies(data, gold_answers_key='answers'):
    """Compute accuracy using exact_presence for each example."""
    accuracies = []
    
    for item in data:
        pred = item.get('output', '').strip()
        gold_answers = item.get(gold_answers_key, [])
        
        if not pred or not gold_answers:
            accuracies.append(0)
            continue
        
        # Use exact_presence from nlp_rag_metrics
        is_accurate = 1 if exact_presence(gold_answers, pred) else 0
        accuracies.append(is_accurate)
    
    return accuracies


def create_pairwise_scatterplots(model_data, accuracy_scores, output_dir):
    """Create pairwise scatter plots for accuracy across all models."""
    
    # Filter models that have valid accuracy scores
    available_models = []
    for k in MODEL_FILES.keys():
        if k in model_data and k in accuracy_scores and accuracy_scores[k] is not None:
            if len(accuracy_scores[k]) > 0:
                available_models.append(k)
    
    if len(available_models) < 2:
        print("⚠ Need at least 2 models with accuracy scores for comparison")
        return
    
    # Generate all pairwise combinations
    model_pairs = list(combinations(available_models, 2))
    n_pairs = len(model_pairs)
    
    # Calculate grid dimensions (prefer 2 columns)
    n_cols = 2
    n_rows = (n_pairs + 1) // 2
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 7 * n_rows))
    
    # Flatten axes array for easy indexing
    if n_pairs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Create scatter plots for each pair
    for idx, (model1, model2) in enumerate(model_pairs):
        ax = axes[idx]
        
        # Get accuracy scores for both models (already 0 or 1)
        # Add small jitter for visualization since values are discrete
        n_examples = min(len(accuracy_scores[model1]), len(accuracy_scores[model2]))
        np.random.seed(42)  # For reproducibility
        jitter = 0.05
        
        acc1 = np.array(accuracy_scores[model1][:n_examples]) + np.random.uniform(-jitter, jitter, n_examples)
        acc2 = np.array(accuracy_scores[model2][:n_examples]) + np.random.uniform(-jitter, jitter, n_examples)
        
        # Create scatter plot with transparency to show density
        ax.scatter(acc1, acc2, alpha=0.3, s=15, color=COLORS[0])
        
        # Add grid lines at 0 and 1
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.3)
        
        # Calculate agreement metrics
        acc1_binary = np.array(accuracy_scores[model1][:n_examples])
        acc2_binary = np.array(accuracy_scores[model2][:n_examples])
        
        both_correct = np.sum((acc1_binary == 1) & (acc2_binary == 1))
        both_wrong = np.sum((acc1_binary == 0) & (acc2_binary == 0))
        model1_only = np.sum((acc1_binary == 1) & (acc2_binary == 0))
        model2_only = np.sum((acc1_binary == 0) & (acc2_binary == 1))
        agreement = (both_correct + both_wrong) / n_examples
        
        # Add labels and title
        ax.set_xlabel(f'{MODEL_NAMES[model1]} Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{MODEL_NAMES[model2]} Accuracy', fontsize=12, fontweight='bold')
        title = f'{MODEL_NAMES[model1]} vs {MODEL_NAMES[model2]}\n'
        title += f'Agreement: {agreement:.1%} | Both Correct: {both_correct:,} | Both Wrong: {both_wrong:,}\n'
        title += f'{MODEL_NAMES[model1]} Only: {model1_only:,} | {MODEL_NAMES[model2]} Only: {model2_only:,}'
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        
        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
    
    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'pairwise_accuracy_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved pairwise accuracy comparison plot: {output_path}")
    plt.close()


def create_agreement_matrix(accuracy_scores, output_dir):
    """Create an agreement matrix heatmap for accuracies."""
    
    # Filter models with valid accuracy scores
    available_models = []
    for k in MODEL_FILES.keys():
        if k in accuracy_scores and accuracy_scores[k] is not None:
            if len(accuracy_scores[k]) > 0:
                available_models.append(k)
    
    if len(available_models) < 2:
        print("⚠ Need at least 2 models for agreement matrix")
        return
    
    # Build agreement matrix
    n_models = len(available_models)
    agreement_matrix = np.zeros((n_models, n_models))
    
    for i, model1 in enumerate(available_models):
        for j, model2 in enumerate(available_models):
            if i == j:
                agreement_matrix[i, j] = 1.0
            else:
                # Calculate agreement percentage
                n_examples = min(len(accuracy_scores[model1]), len(accuracy_scores[model2]))
                acc1 = np.array(accuracy_scores[model1][:n_examples])
                acc2 = np.array(accuracy_scores[model2][:n_examples])
                
                agreement = np.sum(acc1 == acc2) / n_examples
                agreement_matrix[i, j] = agreement
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    labels = [MODEL_NAMES[m] for m in available_models]
    
    sns.heatmap(agreement_matrix, annot=True, fmt='.1%', cmap='RdYlGn', 
                xticklabels=labels, yticklabels=labels,
                vmin=0, vmax=1, square=True, cbar_kws={'label': 'Agreement Rate'},
                ax=ax, linewidths=1, linecolor='white')
    
    ax.set_title('Accuracy Agreement Matrix Across RAG Architectures', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'accuracy_agreement_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved agreement matrix: {output_path}")
    plt.close()


def generate_summary_statistics(accuracy_scores, output_dir):
    """Generate summary statistics for accuracies."""
    
    # Filter models with valid accuracy scores
    available_models = []
    for k in MODEL_FILES.keys():
        if k in accuracy_scores and accuracy_scores[k] is not None:
            if len(accuracy_scores[k]) > 0:
                available_models.append(k)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ACCURACY EDA: PAIRWISE COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Individual model statistics
    report_lines.append("Model Accuracy Statistics:")
    report_lines.append("-" * 80)
    for model in available_models:
        acc_array = np.array(accuracy_scores[model])
        accuracy = acc_array.mean()
        n_correct = acc_array.sum()
        n_total = len(acc_array)
        
        report_lines.append(f"\n{MODEL_NAMES[model]}:")
        report_lines.append(f"  Accuracy:      {accuracy:.1%}")
        report_lines.append(f"  Correct:       {int(n_correct):,} / {n_total:,}")
        report_lines.append(f"  Incorrect:     {int(n_total - n_correct):,}")
    
    # Statistical Significance Testing
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("Statistical Significance Testing:")
    report_lines.append("-" * 80)
    
    if len(available_models) >= 2:
        # Friedman test (if 3+ models) - non-parametric repeated measures test
        if len(available_models) >= 3:
            report_lines.append("\nFriedman Test (Non-Parametric Repeated Measures):")
            report_lines.append("-" * 40)
            report_lines.append("Null Hypothesis: All models have equal accuracy distributions")
            report_lines.append("Alternative: At least one model differs significantly")
            report_lines.append("Note: Appropriate for paired binary data (same examples across models)")
            report_lines.append("")
            
            # Ensure all samples have the same length
            min_length = min(len(accuracy_scores[m]) for m in available_models)
            samples = [accuracy_scores[m][:min_length] for m in available_models]
            
            # Friedman test expects data in shape (n_samples, n_conditions)
            data_matrix = np.column_stack(samples)
            statistic, p_value = stats.friedmanchisquare(*samples)
            
            report_lines.append(f"  χ² statistic:  {statistic:.4f}")
            report_lines.append(f"  p-value:       {p_value:.6f}")
            report_lines.append(f"  # examples:    {min_length:,}")
            report_lines.append(f"  # models:      {len(available_models)}")
            
            if p_value < 0.001:
                report_lines.append(f"  Result:        *** Highly significant (p < 0.001)")
            elif p_value < 0.01:
                report_lines.append(f"  Result:        ** Very significant (p < 0.01)")
            elif p_value < 0.05:
                report_lines.append(f"  Result:        * Significant (p < 0.05)")
            else:
                report_lines.append(f"  Result:        Not significant (p ≥ 0.05)")
            
            report_lines.append("")
            if p_value < 0.05:
                report_lines.append("  ⚠ Friedman test shows significant differences exist between models.")
                report_lines.append("     See pairwise McNemar's tests below for specific comparisons.")
            else:
                report_lines.append("  ℹ Friedman test shows no significant differences between models overall.")
        
        # Pairwise tests (McNemar's test for paired binary data)
        report_lines.append("")
        report_lines.append("\nPairwise Statistical Tests (McNemar's Test):")
        report_lines.append("-" * 40)
        report_lines.append("McNemar's test for paired binary classifications")
        report_lines.append("Null Hypothesis: The two models have equal accuracy")
        report_lines.append("")
        
        model_pairs = list(combinations(available_models, 2))
        for model1, model2 in model_pairs:
            n_examples = min(len(accuracy_scores[model1]), len(accuracy_scores[model2]))
            acc1 = np.array(accuracy_scores[model1][:n_examples])
            acc2 = np.array(accuracy_scores[model2][:n_examples])
            
            # Create contingency table for McNemar's test
            # [model1 correct, model2 wrong] vs [model1 wrong, model2 correct]
            both_correct = np.sum((acc1 == 1) & (acc2 == 1))
            both_wrong = np.sum((acc1 == 0) & (acc2 == 0))
            model1_only = np.sum((acc1 == 1) & (acc2 == 0))
            model2_only = np.sum((acc1 == 0) & (acc2 == 1))
            
            # McNemar's test uses the off-diagonal elements
            # Continuity correction for better approximation with small counts
            if model1_only + model2_only > 0:
                contingency_table = np.array([[both_correct, model1_only], 
                                              [model2_only, both_wrong]])
                result = mcnemar(contingency_table, correction=True)
                statistic = result.statistic
                p_value = result.pvalue
            else:
                # If both off-diagonal elements are 0, models are identical
                statistic = 0.0
                p_value = 1.0
            
            report_lines.append(f"\n{MODEL_NAMES[model1]} vs {MODEL_NAMES[model2]}:")
            report_lines.append(f"  Contingency Table:")
            report_lines.append(f"    Both Correct:  {both_correct:,}")
            report_lines.append(f"    Both Wrong:    {both_wrong:,}")
            report_lines.append(f"    {MODEL_NAMES[model1]} Only:  {model1_only:,}")
            report_lines.append(f"    {MODEL_NAMES[model2]} Only:  {model2_only:,}")
            report_lines.append(f"  McNemar's χ²:   {statistic:.4f}")
            report_lines.append(f"  p-value:        {p_value:.6f}")
            
            if p_value < 0.001:
                report_lines.append(f"  Result:         *** Highly significant difference (p < 0.001)")
            elif p_value < 0.01:
                report_lines.append(f"  Result:         ** Very significant difference (p < 0.01)")
            elif p_value < 0.05:
                report_lines.append(f"  Result:         * Significant difference (p < 0.05)")
            else:
                report_lines.append(f"  Result:         No significant difference (p ≥ 0.05)")
            
            # Interpretation
            if p_value < 0.05:
                if model1_only > model2_only:
                    report_lines.append(f"  ⚠ {MODEL_NAMES[model1]} performs significantly better")
                else:
                    report_lines.append(f"  ⚠ {MODEL_NAMES[model2]} performs significantly better")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("Pairwise Agreement Analysis:")
    report_lines.append("-" * 80)
    
    # Pairwise agreement
    model_pairs = list(combinations(available_models, 2))
    for model1, model2 in model_pairs:
        n_examples = min(len(accuracy_scores[model1]), len(accuracy_scores[model2]))
        acc1 = np.array(accuracy_scores[model1][:n_examples])
        acc2 = np.array(accuracy_scores[model2][:n_examples])
        
        both_correct = np.sum((acc1 == 1) & (acc2 == 1))
        both_wrong = np.sum((acc1 == 0) & (acc2 == 0))
        model1_only = np.sum((acc1 == 1) & (acc2 == 0))
        model2_only = np.sum((acc1 == 0) & (acc2 == 1))
        agreement = (both_correct + both_wrong) / n_examples
        
        report_lines.append(f"\n{MODEL_NAMES[model1]:15} vs {MODEL_NAMES[model2]:15}:")
        report_lines.append(f"  Agreement:       {agreement:.1%}")
        report_lines.append(f"  Both Correct:    {both_correct:,}")
        report_lines.append(f"  Both Wrong:      {both_wrong:,}")
        report_lines.append(f"  {MODEL_NAMES[model1]:15} Only: {model1_only:,}")
        report_lines.append(f"  {MODEL_NAMES[model2]:15} Only: {model2_only:,}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Write to file
    report_path = output_dir / 'accuracy_eda_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Saved EDA report: {report_path}")
    
    # Print to console
    print("\n" + "\n".join(report_lines))


def main():
    print("=" * 80)
    print("ACCURACY EDA: PAIRWISE ANALYSIS")
    print("=" * 80)
    print()
    
    # Load all model data
    print("📂 Loading model results...")
    model_data = {}
    for model_key in MODEL_FILES.keys():
        print(f"  Loading {MODEL_NAMES[model_key]}...")
        data = load_model_data(model_key)
        if data is not None:
            model_data[model_key] = data
            print(f"    ✓ Loaded {len(data):,} examples")
    
    if len(model_data) < 2:
        print("\n⚠ Error: Need at least 2 models for pairwise comparison")
        return
    
    print()
    
    # Compute accuracies for all models
    print("🔢 Computing accuracies...")
    accuracy_scores = {}
    for model_key, data in model_data.items():
        print(f"  Processing {MODEL_NAMES[model_key]}...")
        scores = compute_accuracies(data)
        accuracy_scores[model_key] = scores
        n_correct = sum(scores)
        accuracy = n_correct / len(scores) * 100
        print(f"    ✓ Accuracy: {accuracy:.1f}% ({n_correct:,}/{len(scores):,})")
    
    print()
    
    # Create visualizations
    print("📊 Creating pairwise scatter plots...")
    create_pairwise_scatterplots(model_data, accuracy_scores, OUTPUT_DIR)
    
    print()
    print("📊 Creating agreement matrix...")
    create_agreement_matrix(accuracy_scores, OUTPUT_DIR)
    
    print()
    print("📄 Generating summary statistics...")
    generate_summary_statistics(accuracy_scores, OUTPUT_DIR)
    
    print()
    print("=" * 80)
    print("✓ EDA COMPLETE")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print(f"  - pairwise_accuracy_comparison.png: Scatter plots for all model pairs")
    print(f"  - accuracy_agreement_matrix.png: Agreement heatmap")
    print(f"  - accuracy_eda_report.txt: Statistical summary")
    print()


if __name__ == "__main__":
    main()
