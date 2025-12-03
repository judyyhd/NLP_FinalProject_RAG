#!/usr/bin/env python3
"""
Comprehensive evaluation for RAG systems: No RAG, Vanilla RAG, and InstructRAG.
Evaluates all models together with:
- Semantic similarity
- Response length/verbosity  
- Answer quality assessment
- Comparative analysis
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import re
import string

# Import updated metrics from retrieval_lm_core
sys.path.append('models/retrieval_lm_core')
from metrics import exact_match_score, qa_f1_score, metric_max_over_ground_truths

# Import accuracy calculation from teammate's script
sys.path.append('.')
from nlp_rag_metrics import exact_presence

# Set seaborn style and color palette
sns.set_palette("husl", 10)
COLORS = sns.color_palette("husl", 10)

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = Path("evaluation/results")
OUTPUT_DIR = Path("evaluation/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAMES = {
    'no_rag': 'No RAG',
    'vanilla_rag': 'Vanilla RAG', 
    'instructrag': 'InstructRAG',
    'selfrag': 'Self-RAG'
}

# Lazy load sentence transformer
_sentence_model = None

def get_sentence_model():
    """Lazy load sentence transformer model."""
    global _sentence_model
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading sentence transformer model...")
            _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Model loaded")
        except ImportError:
            print("⚠ sentence-transformers not installed. Semantic similarity will be skipped.")
            _sentence_model = False
    return _sentence_model if _sentence_model else None

# ============================================================================
# Core Metrics
# ============================================================================

def calculate_semantic_similarity(text1, text2):
    """Calculate cosine similarity between two texts using sentence embeddings."""
    model = get_sentence_model()
    if model is None:
        return None
    
    embeddings = model.encode([text1, text2])
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(similarity)

def analyze_verbosity(output):
    """Analyze output verbosity."""
    words = output.split()
    return len(words)

def load_model_data(filepath):
    """Load JSONL results file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

# ============================================================================
# Multi-Model Analysis
# ============================================================================

def analyze_all_models(models_data):
    """Analyze all models together."""
    
    results = {
        'n_examples': len(next(iter(models_data.values()))),
        'models': {}
    }
    
    # Process each model
    for model_name, data in models_data.items():
        print(f"  Processing {MODEL_NAMES.get(model_name, model_name)}...")
        
        em_scores = []
        f1_scores = []
        semantic_scores = []
        response_lengths = []
        accuracy_scores = []
        
        for i, item in enumerate(data):
            if i % 1000 == 0 and i > 0:
                print(f"    Processed {i}/{len(data)} examples...")
            
            output = item.get('output', item.get('prediction', ''))
            answers = item.get('answers', [])
            gold_answer = answers[0] if isinstance(answers, list) and answers else item.get('answer', '')
            
            # Recalculate EM and F1 with updated normalization (teammate's method)
            em_score = metric_max_over_ground_truths(exact_match_score, output, answers)
            f1_score = metric_max_over_ground_truths(qa_f1_score, output, answers)
            
            # Calculate accuracy using teammate's exact_presence method
            accuracy = 1 if exact_presence(answers, output) else 0
            
            em_scores.append(em_score)
            f1_scores.append(f1_score)
            accuracy_scores.append(accuracy)
            response_lengths.append(analyze_verbosity(output))
            
            # Semantic similarity - COMMENTED OUT TO SAVE TIME
            # sem_score = calculate_semantic_similarity(gold_answer, output)
            # if sem_score is not None:
            #     semantic_scores.append(sem_score)
        
        results['models'][model_name] = {
            'em': em_scores,
            'f1': f1_scores,
            'accuracy': accuracy_scores,
            'semantic': semantic_scores if semantic_scores else None,
            'lengths': response_lengths
        }
        
        print(f"    ✓ EM={np.mean(em_scores)*100:.2f}%, F1={np.mean(f1_scores)*100:.2f}%, "
              f"Accuracy={np.mean(accuracy_scores)*100:.2f}%, Avg Length={np.mean(response_lengths):.1f} words")
    
    return results

# ============================================================================
# Unified Visualizations
# ============================================================================

def plot_core_metrics(results, output_dir):
    """Create core metrics bar chart with error bars."""
    
    model_order = ['no_rag', 'vanilla_rag', 'instructrag', 'selfrag']
    available_models = [m for m in model_order if m in results['models']]
    labels = [MODEL_NAMES.get(m, m) for m in available_models]
    
    # Calculate means and stds
    em_means = [np.mean(results['models'][m]['em']) * 100 for m in available_models]
    em_stds = [np.std(results['models'][m]['em']) * 100 for m in available_models]
    f1_means = [np.mean(results['models'][m]['f1']) * 100 for m in available_models]
    f1_stds = [np.std(results['models'][m]['f1']) * 100 for m in available_models]
    accuracy_means = [np.mean(results['models'][m]['accuracy']) * 100 for m in available_models]
    accuracy_stds = [np.std(results['models'][m]['accuracy']) * 100 for m in available_models]
    
    # Create single figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(available_models))
    width = 0.25
    
    # Error bar style - dotted red line with low opacity
    error_kw = {'ecolor': 'red', 'linestyle': ':', 'alpha': 0.4, 'capsize': 5, 'capthick': 1.5, 'elinewidth': 1.5}
    
    # Core metrics bars with error bars
    bars1 = ax.bar(x - width, em_means, width, yerr=em_stds, error_kw=error_kw,
                   label='Exact Match', color=COLORS[0], alpha=0.9)
    bars2 = ax.bar(x, f1_means, width, yerr=f1_stds, error_kw=error_kw,
                   label='F1 Score', color=COLORS[1], alpha=0.9)
    bars3 = ax.bar(x + width, accuracy_means, width, yerr=accuracy_stds, error_kw=error_kw,
                   label='Accuracy', color=COLORS[3], alpha=0.9)
    
    ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Core Metrics: EM, F1, and Accuracy (n={results["n_examples"]:,})', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.legend(fontsize=12, loc='upper right', bbox_to_anchor=(1.2, 1), frameon=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'core_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved core metrics plot")
    plt.close()

def plot_response_length(results, output_dir):
    """Create response length ridge plot."""
    
    model_order = ['no_rag', 'vanilla_rag', 'instructrag', 'selfrag']
    available_models = [m for m in model_order if m in results['models']]
    labels = [MODEL_NAMES.get(m, m) for m in available_models]
    
    # Create figure with subplots for ridge plot effect
    n_models = len(available_models)
    fig, axes = plt.subplots(n_models, 1, figsize=(12, 2 * n_models), sharex=True)
    
    if n_models == 1:
        axes = [axes]
    
    # Create ridge plot - each model gets its own row
    for i, (model, ax) in enumerate(zip(available_models, axes)):
        lengths = results['models'][model]['lengths']
        
        # Create histogram
        counts, bins, patches = ax.hist(lengths, bins=50, range=(0, 50), 
                                         color=COLORS[i], alpha=0.7, edgecolor=COLORS[i], linewidth=1.5)
        
        # Add title inside the plot area (top right) with colored background
        ax.text(0.98, 0.95, labels[i], transform=ax.transAxes, 
                fontsize=13, fontweight='bold', va='top', ha='right', color='black',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS[i], edgecolor=COLORS[i], 
                         linewidth=2, alpha=0.9))
        ax.grid(axis='y', alpha=0.3)
        
        # Keep only left and bottom spines (axes)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
    
    # Only show x-label on bottom plot
    axes[-1].set_xlabel('Word Count', fontsize=13, fontweight='bold')
    
    # Add main title
    fig.suptitle('Response Length Distribution', fontsize=16, fontweight='bold', y=0.995)
    
    # Add shared y-axis label before tight_layout with proper positioning
    fig.supylabel('Frequency', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    plt.savefig(output_dir / 'response_length.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved response length plot")
    plt.close()

def plot_semantic_similarity(results, output_dir):
    """Create semantic similarity violin plot."""
    
    model_order = ['no_rag', 'vanilla_rag', 'instructrag', 'selfrag']
    available_models = [m for m in model_order if m in results['models']]
    labels = [MODEL_NAMES.get(m, m) for m in available_models]
    
    # Check if semantic similarity is available
    has_semantic = all(results['models'][m]['semantic'] is not None for m in available_models)
    
    if not has_semantic:
        print("⚠ Semantic similarity not available, skipping violin plot")
        return
    
    # Create single figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for violin plot (convert to percentages)
    semantic_data = [np.array(results['models'][m]['semantic']) * 100 for m in available_models]
    
    # Create violin plot
    parts = ax.violinplot(semantic_data, positions=range(len(available_models)), 
                          showmeans=True, showmedians=True)
    
    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(COLORS[i])
        pc.set_alpha(0.7)
    
    ax.set_ylabel('Semantic Similarity (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Semantic Similarity Distribution (n={results["n_examples"]:,})', fontsize=16, fontweight='bold')
    ax.set_xticks(range(len(available_models)))
    ax.set_xticklabels(labels, fontsize=13)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add legend manually - outside plot area
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS[i], alpha=0.7, label=labels[i]) 
                      for i in range(len(available_models))]
    ax.legend(handles=legend_elements, fontsize=11, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'semantic_similarity.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved semantic similarity violin plot")
    plt.close()

# ============================================================================
# Unified Report
# ============================================================================

def generate_comprehensive_report(results, output_dir):
    """Generate unified report for all models."""
    
    output_file = output_dir / 'comprehensive_evaluation_report.txt'
    
    model_order = ['no_rag', 'vanilla_rag', 'instructrag', 'selfrag']
    available_models = [m for m in model_order if m in results['models']]
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE EVALUATION REPORT\n")
        f.write(f"Dataset: HotpotQA Full Dev Set ({results['n_examples']:,} examples)\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary table
        f.write("PERFORMANCE SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<20} {'EM (%)':<10} {'F1 (%)':<10} {'Acc (%)':<10} {'Length':<10}")
        
        has_semantic = all(results['models'][m]['semantic'] is not None for m in available_models)
        if has_semantic:
            f.write(f" {'Semantic (%)':<12}")
        f.write("\n")
        f.write("-" * 80 + "\n")
        
        for model in available_models:
            label = MODEL_NAMES.get(model, model)
            em = np.mean(results['models'][model]['em']) * 100
            f1 = np.mean(results['models'][model]['f1']) * 100
            accuracy = np.mean(results['models'][model]['accuracy']) * 100
            length = np.mean(results['models'][model]['lengths'])
            
            f.write(f"{label:<20} {em:<10.2f} {f1:<10.2f} {accuracy:<10.2f} {length:<10.1f}")
            
            if has_semantic:
                semantic = np.mean(results['models'][model]['semantic']) * 100
                f.write(f" {semantic:<12.2f}")
            f.write("\n")
        
        # Detailed statistics
        f.write("\n" + "=" * 80 + "\n\n")
        f.write("DETAILED STATISTICS\n")
        f.write("-" * 80 + "\n\n")
        
        for model in available_models:
            label = MODEL_NAMES.get(model, model)
            data = results['models'][model]
            
            f.write(f"{label}:\n")
            f.write(f"  EM:  mean={np.mean(data['em'])*100:.4f}%, std={np.std(data['em'])*100:.4f}%, "
                   f"min={np.min(data['em'])*100:.4f}%, max={np.max(data['em'])*100:.4f}%\n")
            f.write(f"  F1:  mean={np.mean(data['f1'])*100:.4f}%, std={np.std(data['f1'])*100:.4f}%, "
                   f"min={np.min(data['f1'])*100:.4f}%, max={np.max(data['f1'])*100:.4f}%\n")
            f.write(f"  Accuracy: mean={np.mean(data['accuracy'])*100:.4f}%, std={np.std(data['accuracy'])*100:.4f}%, "
                   f"min={np.min(data['accuracy'])*100:.4f}%, max={np.max(data['accuracy'])*100:.4f}%\n")
            f.write(f"  Response Length: mean={np.mean(data['lengths']):.2f} words, "
                   f"std={np.std(data['lengths']):.2f}, "
                   f"min={np.min(data['lengths']):.0f}, max={np.max(data['lengths']):.0f}\n")
            
            if has_semantic and data['semantic']:
                f.write(f"  Semantic: mean={np.mean(data['semantic'])*100:.4f}%, "
                       f"std={np.std(data['semantic'])*100:.4f}%, "
                       f"min={np.min(data['semantic'])*100:.4f}%, "
                       f"max={np.max(data['semantic'])*100:.4f}%\n")
            
            f.write(f"  Questions with EM=1: {sum(data['em'])} ({sum(data['em'])/len(data['em'])*100:.2f}%)\n")
            f.write(f"  Questions with F1>0.5: {sum(1 for f in data['f1'] if f > 0.5)} "
                   f"({sum(1 for f in data['f1'] if f > 0.5)/len(data['f1'])*100:.2f}%)\n\n")
        
        # Comparative analysis
        if len(available_models) >= 2:
            f.write("=" * 80 + "\n\n")
            f.write("COMPARATIVE ANALYSIS\n")
            f.write("-" * 80 + "\n\n")
            
            baseline_model = 'no_rag'
            if baseline_model in available_models:
                baseline_em = np.mean(results['models'][baseline_model]['em'])
                baseline_f1 = np.mean(results['models'][baseline_model]['f1'])
                baseline_length = np.mean(results['models'][baseline_model]['lengths'])
                if has_semantic:
                    baseline_semantic = np.mean(results['models'][baseline_model]['semantic'])
                
                for model in available_models:
                    if model == baseline_model:
                        continue
                    
                    label = MODEL_NAMES.get(model, model)
                    em_diff = (np.mean(results['models'][model]['em']) - baseline_em) * 100
                    f1_diff = (np.mean(results['models'][model]['f1']) - baseline_f1) * 100
                    length_diff = np.mean(results['models'][model]['lengths']) - baseline_length
                    
                    f.write(f"{label} vs No RAG:\n")
                    f.write(f"  EM improvement: {em_diff:+.4f} percentage points\n")
                    f.write(f"  F1 improvement: {f1_diff:+.4f} percentage points\n")
                    f.write(f"  Response length change: {length_diff:+.2f} words\n")
                    
                    if has_semantic:
                        semantic_diff = (np.mean(results['models'][model]['semantic']) - baseline_semantic) * 100
                        f.write(f"  Semantic improvement: {semantic_diff:+.4f} percentage points\n")
                    f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("Notes:\n")
        f.write("- EM (Exact Match): Strict string match after normalization\n")
        f.write("- F1: Token-level overlap between prediction and ground truth\n")
        f.write("- Length: Average number of words in model predictions\n")
        if has_semantic:
            f.write("- Semantic Similarity: Cosine similarity using sentence embeddings (all-MiniLM-L6-v2)\n")
        f.write(f"- All models evaluated on the same {results['n_examples']:,} dev examples\n")
    
    print(f"✓ Saved comprehensive report")


def find_characteristic_examples(models_data, example_metrics, available_models, n_examples):
    """Find examples that best showcase the differences between all models."""
    
    characteristic_scores = []
    
    for i in range(n_examples):
        # Calculate diversity metrics for this example
        accuracies = [example_metrics[m]['accuracy'][i] for m in available_models]
        f1_scores = [example_metrics[m]['f1'][i] for m in available_models]
        lengths = [example_metrics[m]['lengths'][i] for m in available_models]
        
        # Skip if any model has missing data
        if any(i >= len(example_metrics[m]['accuracy']) for m in available_models):
            continue
        
        # Calculate diversity score (higher = more interesting)
        # 1. Performance diversity - variance in accuracy and F1
        acc_variance = np.var(accuracies)
        f1_variance = np.var(f1_scores)
        
        # 2. Response length diversity - variance in lengths
        length_variance = np.var(lengths)
        
        # 3. Mixed results (not all correct or all wrong)
        acc_sum = sum(accuracies)
        mixed_score = 1.0 if 0 < acc_sum < len(available_models) else 0.5
        
        # 4. Prefer examples with at least one correct answer
        has_correct = 1.0 if acc_sum > 0 else 0.3
        
        # Combined characteristic score
        characteristic_score = (
            acc_variance * 2.0 +          # Weight accuracy diversity heavily
            f1_variance * 1.5 +            # F1 diversity
            length_variance * 0.001 +      # Length diversity (scaled down)
            mixed_score * 1.0 +            # Mixed results bonus
            has_correct * 0.5              # Prefer solvable questions
        )
        
        characteristic_scores.append((i, characteristic_score, accuracies, f1_scores))
    
    # Sort by characteristic score (most diverse first)
    characteristic_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return indices of top characteristic examples
    return [idx for idx, score, acc, f1 in characteristic_scores[:10]]


def generate_example_showcases(models_data, results, output_dir):
    """Generate examples showcasing strengths and weaknesses of each architecture."""
    
    output_file = output_dir / 'example_showcases.txt'
    
    model_order = ['no_rag', 'vanilla_rag', 'instructrag', 'selfrag']
    available_models = [m for m in model_order if m in results['models']]
    
    if len(available_models) < 2:
        print("⚠ Need at least 2 models for example comparison")
        return
    
    # Get per-example metrics for all models
    n_examples = results['n_examples']
    example_metrics = {model: {
        'em': results['models'][model]['em'],
        'f1': results['models'][model]['f1'],
        'accuracy': results['models'][model]['accuracy'],
        'lengths': results['models'][model]['lengths'],
        'semantic': results['models'][model]['semantic']
    } for model in available_models}
    
    with open(output_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("EXAMPLE SHOWCASES: RAG ARCHITECTURE COMPARISON\n")
        f.write("=" * 100 + "\n\n")
        
        # For each model, find examples where it excels and where it fails
        for model in available_models:
            model_name = MODEL_NAMES.get(model, model)
            model_data = models_data[model]
            
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"{model_name.upper()} - STRENGTHS AND WEAKNESSES\n")
            f.write("=" * 100 + "\n\n")
            
            # Find best cases: where this model succeeds but others fail
            f.write(f"--- {model_name} STRENGTHS (Correct when others fail) ---\n\n")
            
            success_examples = []
            for i in range(min(n_examples, len(model_data))):
                this_accuracy = example_metrics[model]['accuracy'][i]
                
                # Check if this model succeeded (accuracy = 1)
                if this_accuracy == 1:
                    # Check how many other models failed
                    other_failures = sum(1 for m in available_models 
                                       if m != model and example_metrics[m]['accuracy'][i] == 0)
                    
                    if other_failures > 0:
                        success_examples.append((i, other_failures))
            
            # Sort by number of other failures
            success_examples.sort(key=lambda x: x[1], reverse=True)
            
            # Show top 2 examples
            for idx, (ex_idx, n_failures) in enumerate(success_examples[:2], 1):
                item = model_data[ex_idx]
                f.write(f"Example {idx}:\n")
                f.write(f"Question: {item.get('question', 'N/A')}\n")
                f.write(f"Gold Answer: {item.get('answers', ['N/A'])}\n\n")
                
                this_acc = example_metrics[model]['accuracy'][ex_idx]
                this_f1 = example_metrics[model]['f1'][ex_idx]
                f.write(f"{model_name} Response (Accuracy={this_acc:.0f}, F1={this_f1:.2f}):\n")
                f.write(f"  {item.get('output', 'N/A')}\n\n")
                
                # Show other models' responses
                for other_model in available_models:
                    if other_model != model:
                        other_data = models_data[other_model]
                        if ex_idx < len(other_data):
                            other_item = other_data[ex_idx]
                            other_acc = example_metrics[other_model]['accuracy'][ex_idx]
                            other_f1 = example_metrics[other_model]['f1'][ex_idx]
                            other_name = MODEL_NAMES.get(other_model, other_model)
                            f.write(f"{other_name} Response (Accuracy={other_acc:.0f}, F1={other_f1:.2f}):\n")
                            f.write(f"  {other_item.get('output', 'N/A')}\n\n")
                
                f.write("-" * 100 + "\n\n")
            
            # Find weakness cases: where this model fails but others succeed
            f.write(f"--- {model_name} WEAKNESSES (Incorrect when others succeed) ---\n\n")
            
            failure_examples = []
            for i in range(min(n_examples, len(model_data))):
                this_accuracy = example_metrics[model]['accuracy'][i]
                
                # Check if this model failed (accuracy = 0)
                if this_accuracy == 0:
                    # Check how many other models succeeded
                    other_successes = sum(1 for m in available_models 
                                        if m != model and example_metrics[m]['accuracy'][i] == 1)
                    
                    if other_successes > 0:
                        failure_examples.append((i, other_successes))
            
            # Sort by number of other successes
            failure_examples.sort(key=lambda x: x[1], reverse=True)
            
            # Show top 2 examples
            for idx, (ex_idx, n_successes) in enumerate(failure_examples[:2], 1):
                item = model_data[ex_idx]
                f.write(f"Example {idx}:\n")
                f.write(f"Question: {item.get('question', 'N/A')}\n")
                f.write(f"Gold Answer: {item.get('answers', ['N/A'])}\n\n")
                
                this_acc = example_metrics[model]['accuracy'][ex_idx]
                this_f1 = example_metrics[model]['f1'][ex_idx]
                f.write(f"{model_name} Response (Accuracy={this_acc:.0f}, F1={this_f1:.2f}):\n")
                f.write(f"  {item.get('output', 'N/A')}\n\n")
                
                # Show other models' responses
                for other_model in available_models:
                    if other_model != model:
                        other_data = models_data[other_model]
                        if ex_idx < len(other_data):
                            other_item = other_data[ex_idx]
                            other_acc = example_metrics[other_model]['accuracy'][ex_idx]
                            other_f1 = example_metrics[other_model]['f1'][ex_idx]
                            other_name = MODEL_NAMES.get(other_model, other_model)
                            f.write(f"{other_name} Response (Accuracy={other_acc:.0f}, F1={other_f1:.2f}):\n")
                            f.write(f"  {other_item.get('output', 'N/A')}\n\n")
                
                f.write("-" * 100 + "\n\n")
        
        # Verbosity comparison - find examples with large length differences
        f.write("\n" + "=" * 100 + "\n")
        f.write("VERBOSITY COMPARISON\n")
        f.write("=" * 100 + "\n\n")
        
        length_diffs = []
        for i in range(n_examples):
            lengths = [example_metrics[m]['lengths'][i] for m in available_models 
                      if i < len(example_metrics[m]['lengths'])]
            if lengths:
                length_range = max(lengths) - min(lengths)
                if length_range > 10:  # Significant difference
                    length_diffs.append((i, length_range, lengths))
        
        length_diffs.sort(key=lambda x: x[1], reverse=True)
        
        for idx, (ex_idx, length_range, lengths) in enumerate(length_diffs[:3], 1):
            # Get the first available model's data for question/answer
            first_model = available_models[0]
            item = models_data[first_model][ex_idx]
            
            f.write(f"Example {idx} (Length range: {length_range:.0f} words):\n")
            f.write(f"Question: {item.get('question', 'N/A')}\n")
            f.write(f"Gold Answer: {item.get('answers', ['N/A'])}\n\n")
            
            for model in available_models:
                model_data = models_data[model]
                if ex_idx < len(model_data):
                    model_item = model_data[ex_idx]
                    model_name = MODEL_NAMES.get(model, model)
                    word_count = example_metrics[model]['lengths'][ex_idx]
                    accuracy = example_metrics[model]['accuracy'][ex_idx]
                    f1_score = example_metrics[model]['f1'][ex_idx]
                    f.write(f"{model_name} ({word_count:.0f} words, Accuracy={accuracy:.0f}, F1={f1_score:.2f}):\n")
                    f.write(f"  {model_item.get('output', 'N/A')}\n\n")
            
            f.write("-" * 100 + "\n\n")
        
        # Semantic similarity showcase - high semantic but low EM (paraphrased answers)
        has_semantic = all(example_metrics[m]['semantic'] is not None for m in available_models)
        if has_semantic:
            f.write("\n" + "=" * 100 + "\n")
            f.write("PARAPHRASING CASES (High Semantic Similarity, Low Exact Match)\n")
            f.write("=" * 100 + "\n\n")
            
            paraphrase_examples = []
            for i in range(n_examples):
                for model in available_models:
                    if (i < len(example_metrics[model]['semantic']) and 
                        example_metrics[model]['semantic'][i] is not None):
                        
                        em = example_metrics[model]['em'][i]
                        semantic = example_metrics[model]['semantic'][i]
                        
                        # High semantic but low EM indicates paraphrasing
                        if semantic > 0.7 and em < 0.5:
                            paraphrase_examples.append((i, model, semantic, em))
            
            # Sort by semantic similarity
            paraphrase_examples.sort(key=lambda x: x[2], reverse=True)
            
            shown = set()
            count = 0
            for ex_idx, model, semantic, em in paraphrase_examples:
                if ex_idx not in shown and count < 3:
                    shown.add(ex_idx)
                    count += 1
                    
                    model_data = models_data[model]
                    item = model_data[ex_idx]
                    model_name = MODEL_NAMES.get(model, model)
                    
                    f.write(f"Example {count}:\n")
                    f.write(f"Question: {item.get('question', 'N/A')}\n")
                    f.write(f"Gold Answer: {item.get('answers', ['N/A'])}\n\n")
                    f.write(f"{model_name} Response (Semantic={semantic:.2f}, EM={em:.2f}):\n")
                    f.write(f"  {item.get('output', 'N/A')}\n\n")
                    f.write("-" * 100 + "\n\n")
        
        # Add characteristic examples section
        if len(available_models) == 4:
            f.write("\n" + "=" * 100 + "\n")
            f.write("CHARACTERISTIC EXAMPLES: ALL 4 MODELS COMPARED\n")
            f.write("=" * 100 + "\n")
            f.write("Examples showing the distinctive behavior patterns of all four architectures\n")
            f.write("=" * 100 + "\n\n")
            
            characteristic_examples = find_characteristic_examples(models_data, example_metrics, available_models, n_examples)
            
            for idx, ex_idx in enumerate(characteristic_examples[:5], 1):
                first_model = available_models[0]
                item = models_data[first_model][ex_idx]
                
                f.write(f"CHARACTERISTIC EXAMPLE {idx}\n")
                f.write("-" * 100 + "\n")
                f.write(f"Question: {item.get('question', 'N/A')}\n")
                f.write(f"Gold Answer: {item.get('answers', ['N/A'])}\n\n")
                
                # Show metrics summary for this example
                f.write("Performance Summary:\n")
                for model in available_models:
                    model_name = MODEL_NAMES.get(model, model)
                    acc = example_metrics[model]['accuracy'][ex_idx]
                    f1 = example_metrics[model]['f1'][ex_idx]
                    length = example_metrics[model]['lengths'][ex_idx]
                    f.write(f"  {model_name:20s}: Accuracy={acc:.0f}, F1={f1:.2f}, Length={length:.0f} words\n")
                f.write("\n")
                
                # Show all responses
                for model in available_models:
                    model_data = models_data[model]
                    if ex_idx < len(model_data):
                        model_item = model_data[ex_idx]
                        model_name = MODEL_NAMES.get(model, model)
                        f.write(f"{model_name} Response:\n")
                        output = model_item.get('output', 'N/A').strip()
                        # Truncate very long responses for readability
                        if len(output) > 500:
                            output = output[:500] + "... [truncated]"
                        f.write(f"  {output}\n\n")
                
                f.write("=" * 100 + "\n\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("END OF EXAMPLE SHOWCASES\n")
        f.write("=" * 100 + "\n")
    
    print(f"✓ Saved example showcases")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 80)
    print()
    
    # Load all available models
    models_data = {}
    
    model_files = {
        'no_rag': RESULTS_DIR / "no_rag_devfull_results.json",
        'vanilla_rag': RESULTS_DIR / "vanilla_rag_devfull_results.json",
        'instructrag': RESULTS_DIR / "instructrag_llama2_devfull_results.json",
        'selfrag': RESULTS_DIR / "selfrag_llama2_devfull_results.json"
    }
    
    print("📊 Loading model results...")
    for model_name, filepath in model_files.items():
        if filepath.exists():
            models_data[model_name] = load_model_data(filepath)
            print(f"✓ Loaded {MODEL_NAMES.get(model_name, model_name)}: {len(models_data[model_name]):,} examples")
    
    if not models_data:
        print("✗ No model results found in", RESULTS_DIR)
        return
    
    print()
    print("🔍 Analyzing all models...")
    results = analyze_all_models(models_data)
    print()
    
    print("📈 Generating visualizations...")
    plot_core_metrics(results, OUTPUT_DIR)
    plot_response_length(results, OUTPUT_DIR)
    # plot_semantic_similarity(results, OUTPUT_DIR)  # COMMENTED OUT TO SAVE TIME
    
    # Clean up old plots
    old_plots = ['all_models_comparison.png', 'detailed_comparison.png', 'verbosity_analysis.png',
                 'model_comparison.png', 'model_comparison_detailed.png', 'comprehensive_report.txt']
    for old_file in old_plots:
        old_path = OUTPUT_DIR / old_file
        if old_path.exists():
            old_path.unlink()
    print()
    
    print("📄 Generating comprehensive report...")
    generate_comprehensive_report(results, OUTPUT_DIR)
    print()
    
    print("📝 Generating example showcases...")
    generate_example_showcases(models_data, results, OUTPUT_DIR)
    print()
    
    print("=" * 80)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 80)
    print()
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print(f"  - core_metrics.png: EM, F1, and Accuracy with error bars")
    print(f"  - response_length.png: Response length distribution histogram")
    print(f"  - semantic_similarity.png: Semantic similarity violin plot")
    print(f"  - comprehensive_evaluation_report.txt: Complete statistical analysis")
    print(f"  - example_showcases.txt: Example responses showing strengths/weaknesses")
    print()

if __name__ == "__main__":
    main()
