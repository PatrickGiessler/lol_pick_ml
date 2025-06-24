"""
Utility functions for string replacement and template processing
"""

import re
from typing import Dict, Any


def replace_placeholder(template: str, replacements: Dict[str, Any]) -> str:
    """
    Replace placeholders in template strings using ${placeholder} syntax
    
    Args:
        template: Template string with ${placeholder} syntax
        replacements: Dictionary of placeholder -> replacement value
        
    Returns:
        String with placeholders replaced
        
    Example:
        >>> replace_placeholder("Hello ${name}!", {"name": "World"})
        'Hello World!'
        
        >>> replace_placeholder("version_${version}_${date}", {
        ...     "version": "14.1", 
        ...     "date": "2024-01-15"
        ... })
        'version_14.1_2024-01-15'
    """
    result = template
    for key, value in replacements.items():
        placeholder = f"${{{key}}}"
        result = result.replace(placeholder, str(value))
    return result


def replace_placeholder_regex(template: str, replacements: Dict[str, Any]) -> str:
    """
    Replace placeholders using regex for more robust replacement
    
    Args:
        template: Template string with ${placeholder} syntax
        replacements: Dictionary of placeholder -> replacement value
        
    Returns:
        String with placeholders replaced
    """
    def replace_func(match):
        key = match.group(1)
        return str(replacements.get(key, match.group(0)))
    
    # Pattern to match ${placeholder}
    pattern = r'\$\{([^}]+)\}'
    return re.sub(pattern, replace_func, template)


def replace_multiple_templates(templates: Dict[str, str], replacements: Dict[str, Any]) -> Dict[str, str]:
    """
    Replace placeholders in multiple template strings
    
    Args:
        templates: Dictionary of template name -> template string
        replacements: Dictionary of placeholder -> replacement value
        
    Returns:
        Dictionary with all templates processed
    """
    return {
        name: replace_placeholder(template, replacements)
        for name, template in templates.items()
    }


# Example usage and tests
if __name__ == "__main__":
    # Test basic replacement
    template1 = "champions/score_calculator/${version}"
    replacements1 = {"version": "14.1"}
    result1 = replace_placeholder(template1, replacements1)
    print(f"Template: {template1}")
    print(f"Replacements: {replacements1}")
    print(f"Result: {result1}")
    print()
    
    # Test filename replacement
    template2 = "champion_data_${version}_${date}.json"
    replacements2 = {"version": "14.1", "date": "2024-01-15T10:30:00"}
    result2 = replace_placeholder(template2, replacements2)
    print(f"Template: {template2}")
    print(f"Replacements: {replacements2}")
    print(f"Result: {result2}")
    print()
    
    # Test multiple templates
    templates = {
        "score_path": "champions/score_calculator/${version}",
        "file_name": "champion_data_${version}_${date}.json",
        "backup_path": "backups/${version}/${date}"
    }
    replacements = {
        "version": "14.1",
        "date": "2024-01-15"
    }
    results = replace_multiple_templates(templates, replacements)
    print("Multiple template replacement:")
    for name, result in results.items():
        print(f"  {name}: {result}")
    print()
    
    # Test regex version
    template3 = "Hello ${name}, your score is ${score} for version ${version}!"
    replacements3 = {"name": "Player1", "score": 95.5, "version": "14.1"}
    result3 = replace_placeholder_regex(template3, replacements3)
    print(f"Regex replacement result: {result3}")
