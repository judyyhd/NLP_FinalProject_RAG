"""
Color palette for evaluation plots using Seaborn defaults.
"""

import seaborn as sns

# Use seaborn's default color palette
COLORS = sns.color_palette("husl", 8)

# Model-specific color assignments for consistency across plots
MODEL_COLORS = {
    'no_rag': COLORS[0],
    'vanilla_rag': COLORS[1],
    'instructrag': COLORS[2],
    'self_rag': COLORS[3],
    'selfrag': COLORS[3],
    'instructrag_llama2': COLORS[2],
    'instructrag_llama3': COLORS[4],
    'instructrag_ft': COLORS[5],
}

def get_model_color(model_name):
    """Get color for a specific model."""
    # Normalize model name
    normalized = model_name.lower().replace(' ', '_').replace('-', '_')
    
    # Try direct match
    if normalized in MODEL_COLORS:
        return MODEL_COLORS[normalized]
    
    # Try partial matches
    for key, color in MODEL_COLORS.items():
        if key in normalized or normalized in key:
            return MODEL_COLORS[key]
    
    # Default to first color
    return COLORS[0]

def get_comparison_colors(n_models):
    """Get n colors for model comparison plots."""
    return sns.color_palette("husl", n_models)

