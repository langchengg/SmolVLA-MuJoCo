"""
Language Instruction Augmentation for VLA Generalization Testing.

Implements synonym replacement, paraphrase generation, and structured
language variations for testing language-conditioned generalization.
This is a key component of Innovation Point #1.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Synonym Dictionaries ──────────────────────────────────────────────────────

# Action verbs
ACTION_SYNONYMS = {
    "pick up": ["grasp", "grab", "take", "lift", "seize", "clutch", "snatch"],
    "put": ["place", "set", "position", "lay", "deposit", "rest"],
    "push": ["press", "shove", "nudge", "slide", "move"],
    "pull": ["drag", "draw", "tug", "yank"],
    "open": ["unlock", "unfasten", "unseal", "swing open"],
    "close": ["shut", "seal", "fasten", "swing shut"],
    "turn": ["rotate", "twist", "spin", "swivel"],
    "move": ["shift", "transfer", "relocate", "transport", "slide"],
    "stack": ["pile", "heap", "layer", "build up"],
    "flip": ["turn over", "invert", "overturn", "reverse"],
    "pour": ["decant", "empty", "transfer liquid from"],
    "wipe": ["clean", "scrub", "sweep", "rub"],
    "insert": ["put in", "slide in", "place inside", "fit into"],
}

# Color adjectives
COLOR_SYNONYMS = {
    "red": ["crimson", "scarlet", "ruby", "cherry", "vermillion"],
    "blue": ["azure", "cobalt", "navy", "sapphire", "cerulean"],
    "green": ["emerald", "jade", "lime", "olive", "forest green"],
    "yellow": ["golden", "amber", "lemon", "canary", "honey-colored"],
    "white": ["ivory", "pearl", "snow-white", "alabaster"],
    "black": ["onyx", "ebony", "jet-black", "charcoal"],
    "orange": ["tangerine", "amber", "burnt orange", "coral"],
    "purple": ["violet", "plum", "lavender", "magenta"],
    "brown": ["chestnut", "mahogany", "tan", "bronze", "sienna"],
    "pink": ["rose", "fuchsia", "salmon", "blush"],
    "gray": ["silver", "slate", "charcoal", "ash-colored"],
    "grey": ["silver", "slate", "charcoal", "ash-colored"],
}

# Object nouns
OBJECT_SYNONYMS = {
    "cube": ["block", "box", "brick", "square piece"],
    "bowl": ["dish", "container", "basin", "cup"],
    "plate": ["saucer", "dish", "flat surface", "tray"],
    "can": ["tin", "cylinder", "container", "canister"],
    "bottle": ["flask", "container", "vessel", "jug"],
    "mug": ["cup", "drinking vessel", "tankard"],
    "button": ["switch", "control", "knob", "trigger"],
    "drawer": ["compartment", "cabinet drawer", "storage unit"],
    "door": ["panel", "gate", "entrance", "portal"],
    "handle": ["knob", "grip", "lever", "pull"],
    "lid": ["cover", "cap", "top", "seal"],
    "ball": ["sphere", "round object", "orb"],
}

# Spatial prepositions
SPATIAL_SYNONYMS = {
    "on": ["on top of", "upon", "atop", "above"],
    "in": ["inside", "within", "into", "in the interior of"],
    "next to": ["beside", "near", "adjacent to", "alongside"],
    "behind": ["at the back of", "beyond", "past"],
    "in front of": ["before", "ahead of", "facing"],
    "under": ["beneath", "below", "underneath"],
    "between": ["amid", "in the middle of", "among"],
}


@dataclass
class AugmentationConfig:
    """Configuration for language augmentation."""
    synonym_probability: float = 0.5
    paraphrase_probability: float = 0.3
    max_synonyms_per_sentence: int = 3
    seed: int = 42
    templates: list[str] = field(default_factory=lambda: [
        "{action} the {color} {object}",
        "{action} the {object} that is {color}",
        "could you {action} the {color} {object}?",
        "please {action} the {color} {object}",
        "I need you to {action} the {color} {object}",
        "the {color} {object}, {action} it",
    ])


class LanguageAugmentor:
    """
    Language instruction augmentor for testing VLA language generalization.
    
    Generates diverse linguistic variations of task instructions while
    preserving semantic meaning. Used for Innovation Point #1.
    
    Strategies:
        1. Synonym replacement (action verbs, colors, objects)
        2. Template-based paraphrasing
        3. Structural rewriting
        4. Formality variation (casual ↔ formal)
    
    Example:
        >>> aug = LanguageAugmentor()
        >>> variants = aug.generate_variants("pick up the red cube", n=5)
        >>> # ["grasp the crimson block", "grab the scarlet cube", ...]
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        self.rng = random.Random(self.config.seed)
        
        # Build combined synonym lookup
        self._synonyms = {}
        for group in [ACTION_SYNONYMS, COLOR_SYNONYMS, OBJECT_SYNONYMS, SPATIAL_SYNONYMS]:
            self._synonyms.update(group)

    def generate_variants(self, instruction: str, n: int = 5) -> list[str]:
        """
        Generate n diverse variants of the given instruction.
        
        Args:
            instruction: Original language instruction
            n: Number of variants to generate
            
        Returns:
            List of n unique variant instructions
        """
        variants = set()
        max_attempts = n * 10  # Avoid infinite loop
        attempts = 0
        
        while len(variants) < n and attempts < max_attempts:
            attempts += 1
            
            # Randomly choose augmentation strategy
            strategy = self.rng.choice([
                self._synonym_replace,
                self._template_paraphrase,
                self._structural_rewrite,
                self._formality_shift,
            ])
            
            variant = strategy(instruction)
            if variant != instruction and variant not in variants:
                variants.add(variant)
        
        result = sorted(variants)
        
        if len(result) < n:
            logger.warning(
                f"Could only generate {len(result)}/{n} unique variants "
                f"for: '{instruction}'"
            )
        
        return result

    def augment_dataset_instructions(
        self, instructions: list[str], n_variants: int = 3
    ) -> dict[str, list[str]]:
        """
        Generate augmentations for an entire list of instructions.
        
        Returns:
            Dict mapping original instruction → list of variants
        """
        augmented = {}
        for inst in instructions:
            augmented[inst] = self.generate_variants(inst, n=n_variants)
        return augmented

    def _synonym_replace(self, instruction: str) -> str:
        """Replace words/phrases with synonyms."""
        result = instruction.lower()
        replacements_made = 0
        
        # Sort by phrase length (longest first) to handle multi-word phrases
        sorted_phrases = sorted(self._synonyms.keys(), key=len, reverse=True)
        
        for phrase in sorted_phrases:
            if replacements_made >= self.config.max_synonyms_per_sentence:
                break
                
            if phrase in result:
                if self.rng.random() < self.config.synonym_probability:
                    synonyms = self._synonyms[phrase]
                    replacement = self.rng.choice(synonyms)
                    result = result.replace(phrase, replacement, 1)
                    replacements_made += 1
        
        return result

    def _template_paraphrase(self, instruction: str) -> str:
        """Rewrite instruction using a template."""
        # Parse instruction components
        components = self._parse_instruction(instruction)
        
        if not components:
            return self._synonym_replace(instruction)
        
        # Choose a random template
        template = self.rng.choice(self.config.templates)
        
        try:
            result = template.format(**components)
        except KeyError:
            result = self._synonym_replace(instruction)
        
        return result

    def _structural_rewrite(self, instruction: str) -> str:
        """Rewrite the sentence structure while keeping meaning."""
        lower = instruction.lower().strip()
        
        # Active → Passive-ish rewrites
        rewrites = [
            # Imperative → polite request
            (lambda s: s, lambda s: f"please {s}"),
            # Imperative → descriptive
            (lambda s: s, lambda s: f"your task is to {s}"),
            # Add politeness
            (lambda s: s, lambda s: f"could you {s}?"),
            # Add urgency
            (lambda s: s, lambda s: f"I need you to {s}"),
            # Object-fronting
            (lambda s: s, lambda s: self._front_object(s)),
        ]
        
        _, rewriter = self.rng.choice(rewrites)
        return rewriter(lower)

    def _formality_shift(self, instruction: str) -> str:
        """Shift formality level (casual ↔ formal)."""
        lower = instruction.lower().strip()
        
        # Casual transformations
        casual_transforms = {
            "pick up": "grab",
            "place": "put",
            "please ": "",
            "could you ": "",
            "would you ": "",
            "I need you to ": "",
        }
        
        # Formal transformations
        formal_transforms = {
            "grab": "carefully grasp",
            "put": "gently place",
            "get": "retrieve",
            "move": "carefully relocate",
        }
        
        transforms = self.rng.choice([casual_transforms, formal_transforms])
        
        result = lower
        for old, new in transforms.items():
            if old in result:
                result = result.replace(old, new, 1)
                break
        
        return result

    def _parse_instruction(self, instruction: str) -> dict[str, str]:
        """
        Parse instruction into semantic components (action, color, object, etc.)
        Simple rule-based parser.
        """
        lower = instruction.lower().strip()
        components = {}
        
        # Find action
        for action in sorted(ACTION_SYNONYMS.keys(), key=len, reverse=True):
            if lower.startswith(action) or f" {action} " in f" {lower} ":
                components["action"] = action
                break
        
        # Find color
        for color in COLOR_SYNONYMS.keys():
            if color in lower:
                components["color"] = color
                break
        
        # Find object
        for obj in sorted(OBJECT_SYNONYMS.keys(), key=len, reverse=True):
            if obj in lower:
                components["object"] = obj
                break
        
        return components

    def _front_object(self, instruction: str) -> str:
        """Move the object to the front of the sentence."""
        components = self._parse_instruction(instruction)
        
        if "object" in components and "action" in components:
            color = components.get("color", "")
            obj = components["object"]
            action = components["action"]
            
            if color:
                return f"the {color} {obj}, {action} it"
            return f"the {obj}, {action} it"
        
        return instruction

    def get_generalization_matrix(
        self, original_instructions: list[str], n_variants: int = 5
    ) -> dict:
        """
        Generate a full generalization test matrix.
        
        Returns:
            {
                "original": [...],
                "variants": {orig: [v1, v2, ...]},
                "difficulty": {orig: {"synonym": [...], "paraphrase": [...], ...}}
            }
        """
        matrix = {
            "original": original_instructions,
            "variants": {},
            "difficulty_levels": {},
        }
        
        for inst in original_instructions:
            # Generate variants at different difficulty levels
            easy = [self._synonym_replace(inst) for _ in range(n_variants)]
            medium = [self._template_paraphrase(inst) for _ in range(n_variants)]
            hard = [self._structural_rewrite(inst) for _ in range(n_variants)]
            
            # Deduplicate
            easy = list(set(v for v in easy if v != inst))
            medium = list(set(v for v in medium if v != inst and v not in easy))
            hard = list(set(v for v in hard if v != inst and v not in easy and v not in medium))
            
            matrix["variants"][inst] = easy + medium + hard
            matrix["difficulty_levels"][inst] = {
                "easy_synonym": easy[:n_variants],
                "medium_paraphrase": medium[:n_variants],
                "hard_structural": hard[:n_variants],
            }
        
        return matrix

    def __repr__(self):
        n_synonyms = sum(len(v) for v in self._synonyms.values())
        return f"LanguageAugmentor(synonym_groups={len(self._synonyms)}, total_synonyms={n_synonyms})"
