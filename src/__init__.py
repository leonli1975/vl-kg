from .three_layer_kg import ThreeLayerKG, ATOMIC_MOTIFS, COMPOSITE_PATTERNS, LAYOUT_TYPES, TRANSFORMATION_TYPES
# EnhancedThreeLayerKG imported lazily to avoid circular import with understand_image
from .understand_image import ImageUnderstandingPipeline

def get_enhanced_kg():
    from .enhanced_kg import EnhancedThreeLayerKG
    return EnhancedThreeLayerKG

