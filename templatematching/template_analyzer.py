import cv2
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from app.logging_config import get_logger

logger = get_logger(__name__)

class TemplateAnalyzer:
    """Analyzes template quality and suggests improvements"""
    
    def __init__(self):
        self.problem_champions = ['lucian', 'renekton', 'lux', 'seraphine']
    
    def analyze_template_quality(self, template: np.ndarray, champion_name: str) -> Dict:
        """
        Analyze template image quality and suggest improvements
        
        Args:
            template: Template image (BGR or grayscale)
            champion_name: Name of the champion
            
        Returns:
            Dictionary with analysis results and suggestions
        """
        if template is None or template.size == 0:
            return {'error': 'Invalid template'}
        
        # Convert to grayscale if needed
        if len(template.shape) == 3:
            gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            gray = template
            
        analysis = {
            'champion': champion_name,
            'size': template.shape,
            'issues': [],
            'suggestions': [],
            'quality_score': 0.0
        }
        
        # 1. Check for sufficient contrast
        contrast = gray.std()
        analysis['contrast'] = float(contrast)
        if contrast < 20:
            analysis['issues'].append('Low contrast')
            analysis['suggestions'].append('Increase contrast or use different template variant')
        
        # 2. Check for distinctive features (edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        analysis['edge_density'] = float(edge_density)
        if edge_density < 0.1:
            analysis['issues'].append('Few distinctive features')
            analysis['suggestions'].append('Use template with more distinctive visual elements')
        
        # 3. Check for uniform regions (problematic for matching)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        dominant_color_ratio = np.max(hist_norm)
        analysis['dominant_color_ratio'] = float(dominant_color_ratio)
        if dominant_color_ratio > 0.4:
            analysis['issues'].append('Too much uniform color')
            analysis['suggestions'].append('Crop to more varied region or use different template')
        
        # 4. Size analysis
        h, w = gray.shape
        if min(h, w) < 30:
            analysis['issues'].append('Template too small')
            analysis['suggestions'].append('Use larger template for better matching')
        if max(h, w) > 200:
            analysis['issues'].append('Template might be too large')
            analysis['suggestions'].append('Consider smaller crop or lower scale factors')
        
        # 5. Special handling for known problematic champions
        if champion_name.lower() in self.problem_champions:
            analysis['known_issue'] = True
            if champion_name.lower() in ['lucian', 'renekton']:
                analysis['suggestions'].append('Try cropping to face/distinctive armor pieces only')
                analysis['suggestions'].append('Consider multiple template variants (different poses/skins)')
            elif champion_name.lower() in ['lux', 'seraphine']:
                analysis['suggestions'].append('These champions may have generic features - consider stricter thresholds')
                analysis['suggestions'].append('Add negative templates (empty slots) to reduce false positives')
        
        # Calculate overall quality score
        quality_score = 1.0
        if contrast < 20: quality_score -= 0.3
        if edge_density < 0.1: quality_score -= 0.3
        if dominant_color_ratio > 0.4: quality_score -= 0.2
        if min(h, w) < 30: quality_score -= 0.2
        
        analysis['quality_score'] = max(0.0, quality_score)
        
        return analysis
    
    def suggest_scale_factors(self, template_size: Tuple[int, int], target_zone_size: Tuple[int, int]) -> List[float]:
        """
        Suggest appropriate scale factors based on template and zone sizes
        
        Args:
            template_size: (height, width) of template
            target_zone_size: (height, width) of target zone
            
        Returns:
            List of recommended scale factors
        """
        t_h, t_w = template_size
        z_h, z_w = target_zone_size
        
        # Calculate scale needed to fit in zone (with margin)
        scale_w = (z_w * 0.8) / t_w  # 80% of zone width
        scale_h = (z_h * 0.8) / t_h  # 80% of zone height
        
        # Use the more restrictive scale
        max_scale = min(scale_w, scale_h)
        
        # Generate scale factors around the optimal size
        scales = []
        
        # Start with smaller scales for precision
        if max_scale > 0.3:
            scales.extend([0.3, 0.4, 0.5])
        
        # Add scales around the calculated optimal scale
        if max_scale > 0.6:
            base_scale = max_scale * 0.8  # Slightly smaller than max
            scales.extend([
                round(base_scale * 0.8, 2),
                round(base_scale * 0.9, 2),
                round(base_scale, 2),
                round(base_scale * 1.1, 2)
            ])
        
        # Always include 1.0 if reasonable
        if max_scale >= 1.0:
            scales.append(1.0)
        
        # Remove duplicates and sort
        scales = sorted(list(set(scales)))
        
        # Limit to reasonable range
        scales = [s for s in scales if 0.2 <= s <= 2.0]
        
        logger.info(f"Template size: {template_size}, Zone size: {target_zone_size}")
        logger.info(f"Suggested scales: {scales}")
        
        return scales[:6]  # Limit to 6 scales max for performance
    
    def create_template_report(self, template_manager, zones: List) -> str:
        """Create a comprehensive template quality report"""
        report = ["=== TEMPLATE QUALITY ANALYSIS ===\n"]
        
        problem_templates = []
        
        for zone in zones:
            report.append(f"Zone: {zone.label} ({zone.shape.value})")
            report.append(f"Size: {zone.width}x{zone.height}")
            report.append(f"Scale factors: {zone.scale_factor}")
            
            templates = template_manager.get_templates_by_scale_and_shape(zone.scale_factor, zone.shape)
            
            for template in templates:
                analysis = self.analyze_template_quality(template.template, template.name)
                
                if analysis['quality_score'] < 0.7 or analysis.get('known_issue', False):
                    problem_templates.append((template.name, analysis))
                    
            report.append(f"Templates: {len(templates)}")
            report.append("")
        
        if problem_templates:
            report.append("=== PROBLEMATIC TEMPLATES ===")
            for name, analysis in problem_templates:
                report.append(f"\nChampion: {name}")
                report.append(f"Quality Score: {analysis['quality_score']:.2f}")
                if analysis['issues']:
                    report.append(f"Issues: {', '.join(analysis['issues'])}")
                if analysis['suggestions']:
                    report.append("Suggestions:")
                    for suggestion in analysis['suggestions']:
                        report.append(f"  - {suggestion}")
        
        return "\n".join(report)
