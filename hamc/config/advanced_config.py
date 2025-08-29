# hamc/config/advanced_config.py
"""
Advanced Configuration System for HAMC
Centralizes all hardcoded values and provides dynamic configuration loading.
"""

from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path


class ConfigProfile(Enum):
    """Configuration profiles for different environments."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"
    PERFORMANCE = "performance"


@dataclass
class EntropyConfig:
    """Configuration for entropy calculation and adaptive algorithms."""
    
    # Basic entropy parameters
    default_temperature: float = 1.0
    default_learning_rate: float = 0.01
    
    # Temporal decay for pattern learning
    temporal_decay_factor: float = 0.95
    temporal_decay_interval: float = 10.0
    
    # Success rate thresholds
    min_success_rate: float = 0.3
    max_success_rate: float = 0.7
    optimal_success_rate: float = 0.5
    
    # Temperature adjustment factors
    temperature_decay_factor: float = 0.95
    min_temperature: float = 0.5
    max_temperature: float = 2.0
    
    # Learning weights for different levels
    global_learning_weight: float = 0.5
    intermediate_learning_weight: float = 0.3
    local_learning_weight: float = 0.2
    
    # Success score smoothing
    success_smoothing_factor: float = 0.1
    success_decay_factor: float = 0.9
    
    # Trend detection
    trend_threshold: float = 0.1
    trend_window_size: int = 10


@dataclass
class CacheConfig:
    """Configuration for caching systems."""
    
    # Compatibility cache
    compatibility_cache_max_size: int = 1024
    compatibility_cache_ttl: float = 300.0
    
    # Pattern cache
    pattern_cache_max_size: int = 2048
    pattern_cache_ttl: float = 600.0
    
    # Result cache
    result_cache_max_size: int = 512
    result_cache_ttl: float = 1800.0
    
    # Cache performance thresholds
    min_hit_ratio: float = 0.1
    cache_cleanup_interval: int = 100


@dataclass
class GeneratorConfig:
    """Configuration for all generator types."""
    
    # Global generator
    global_min_regions_per_type: int = 1
    global_region_placement_probability: float = 0.9
    global_fallback_probability: float = 0.1
    
    # Local generator - River probabilities
    local_river_water_probability: float = 0.8
    local_river_grass_probability: float = 0.2
    
    # Local generator - Road probabilities
    local_road_asphalt_probability: float = 0.8
    local_road_line_probability: float = 0.2
    
    # Local generator - Road network
    local_road_network_asphalt_probability: float = 0.7
    local_road_network_line_probability: float = 0.3
    
    # Local generator - Desert probabilities
    local_desert_sand_probability: float = 0.9
    local_desert_grass_probability: float = 0.1
    
    # Local generator - Distance calculations
    local_water_probability_decay: float = 0.9
    local_min_water_probability: float = 0.0


@dataclass
class ValidatorConfig:
    """Configuration for validation algorithms."""
    
    # Water validation
    min_water_percentage: float = 0.1
    max_water_percentage: float = 0.8
    
    # Path validation
    min_path_length: int = 3
    max_path_gaps: int = 0
    
    # Compatibility validation
    strict_compatibility: bool = True
    allow_partial_compatibility: bool = False


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    
    # Parallel processing
    max_workers: int = 4
    chunk_size: int = 100
    enable_parallel_processing: bool = True
    
    # Memory management
    max_memory_usage_mb: int = 512
    enable_memory_pool: bool = True
    memory_pool_size: int = 1000
    
    # Profiling
    enable_profiling: bool = False
    profile_output_dir: str = "performance_profiles"
    profile_interval: int = 1000


@dataclass
class LoggingConfig:
    """Configuration for logging system."""
    
    # Log levels
    default_level: str = "INFO"
    file_level: str = "DEBUG"
    console_level: str = "INFO"
    
    # Log files
    log_dir: str = "logs"
    main_log_file: str = "hamc.log"
    error_log_file: str = "hamc_error.log"
    performance_log_file: str = "hamc_performance.log"
    
    # Log format
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_log_size_mb: int = 10
    backup_count: int = 5


@dataclass
class AdvancedConfig:
    """Master configuration class that contains all sub-configurations."""
    
    # Profile management
    active_profile: ConfigProfile = ConfigProfile.DEVELOPMENT
    
    # Sub-configurations
    entropy: EntropyConfig = field(default_factory=EntropyConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    validator: ValidatorConfig = field(default_factory=ValidatorConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Configuration file paths
    config_dir: str = "hamc/config"
    profiles_dir: str = "hamc/config/profiles"
    
    def __post_init__(self):
        """Initialize configuration after creation."""
        self._load_profile_defaults()
    
    def _load_profile_defaults(self):
        """Load default values based on active profile."""
        if self.active_profile == ConfigProfile.PRODUCTION:
            self._apply_production_defaults()
        elif self.active_profile == ConfigProfile.TESTING:
            self._apply_testing_defaults()
        elif self.active_profile == ConfigProfile.PERFORMANCE:
            self._apply_performance_defaults()
    
    def _apply_production_defaults(self):
        """Apply production-optimized defaults."""
        self.logging.default_level = "WARNING"
        self.performance.enable_profiling = False
        self.cache.compatibility_cache_max_size = 2048
        self.performance.max_workers = 8
    
    def _apply_testing_defaults(self):
        """Apply testing-friendly defaults."""
        self.logging.default_level = "DEBUG"
        self.performance.enable_profiling = True
        self.cache.compatibility_cache_max_size = 256
        self.performance.max_workers = 1
    
    def _apply_performance_defaults(self):
        """Apply performance-testing defaults."""
        self.performance.enable_profiling = True
        self.performance.max_workers = 16
        self.cache.compatibility_cache_max_size = 4096
        self.logging.performance_log_file = "hamc_performance_detailed.log"
    
    @classmethod
    def from_file(cls, config_path: str) -> 'AdvancedConfig':
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert profile string to enum
            if 'active_profile' in data:
                data['active_profile'] = ConfigProfile(data['active_profile'])
            
            # Convert nested dictionaries to dataclasses
            if 'entropy' in data and isinstance(data['entropy'], dict):
                data['entropy'] = EntropyConfig(**data['entropy'])
            if 'cache' in data and isinstance(data['cache'], dict):
                data['cache'] = CacheConfig(**data['cache'])
            if 'generator' in data and isinstance(data['generator'], dict):
                data['generator'] = GeneratorConfig(**data['generator'])
            if 'validator' in data and isinstance(data['validator'], dict):
                data['validator'] = ValidatorConfig(**data['validator'])
            if 'performance' in data and isinstance(data['performance'], dict):
                data['performance'] = PerformanceConfig(**data['performance'])
            if 'logging' in data and isinstance(data['logging'], dict):
                data['logging'] = LoggingConfig(**data['logging'])
            
            return cls(**data)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            return cls()
    
    def to_file(self, config_path: str):
        """Save configuration to JSON file."""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Convert to dict for JSON serialization
            data = self.__dict__.copy()
            data['active_profile'] = self.active_profile.value
            
            # Convert dataclasses to dictionaries
            if hasattr(self.entropy, '__dict__'):
                data['entropy'] = self.entropy.__dict__
            if hasattr(self.cache, '__dict__'):
                data['cache'] = self.cache.__dict__
            if hasattr(self.generator, '__dict__'):
                data['generator'] = self.generator.__dict__
            if hasattr(self.validator, '__dict__'):
                data['validator'] = self.validator.__dict__
            if hasattr(self.performance, '__dict__'):
                data['performance'] = self.performance.__dict__
            if hasattr(self.logging, '__dict__'):
                data['logging'] = self.logging.__dict__
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving config to {config_path}: {e}")
    
    def get_nested_value(self, key_path: str) -> Any:
        """Get nested configuration value using dot notation.
        
        Example: 'entropy.default_temperature' or 'cache.compatibility_cache_max_size'
        """
        keys = key_path.split('.')
        value = self
        
        for key in keys:
            if hasattr(value, key):
                value = getattr(value, key)
            else:
                raise KeyError(f"Configuration key '{key}' not found in path '{key_path}'")
        
        return value
    
    def set_nested_value(self, key_path: str, value: Any):
        """Set nested configuration value using dot notation."""
        keys = key_path.split('.')
        obj = self
        
        # Navigate to the parent object
        for key in keys[:-1]:
            if hasattr(obj, key):
                obj = getattr(obj, key)
            else:
                raise KeyError(f"Configuration key '{key}' not found in path '{key_path}'")
        
        # Set the final value
        final_key = keys[-1]
        if hasattr(obj, final_key):
            setattr(obj, final_key, value)
        else:
            raise KeyError(f"Configuration key '{final_key}' not found")


# Global configuration instance
_config_instance: Optional[AdvancedConfig] = None


def get_config() -> AdvancedConfig:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = AdvancedConfig()
    return _config_instance


def load_config(config_path: Optional[str] = None, profile: Optional[ConfigProfile] = None) -> AdvancedConfig:
    """Load configuration from file with optional profile override."""
    global _config_instance
    
    if config_path and os.path.exists(config_path):
        _config_instance = AdvancedConfig.from_file(config_path)
    else:
        _config_instance = AdvancedConfig()
    
    if profile:
        _config_instance.active_profile = profile
        _config_instance._load_profile_defaults()
    
    return _config_instance


def save_config(config_path: str, config: Optional[AdvancedConfig] = None):
    """Save configuration to file."""
    if config is None:
        config = get_config()
    config.to_file(config_path)


# Convenience functions for common configuration access
def get_entropy_config() -> EntropyConfig:
    """Get entropy configuration."""
    return get_config().entropy


def get_cache_config() -> CacheConfig:
    """Get cache configuration."""
    return get_config().cache


def get_generator_config() -> GeneratorConfig:
    """Get generator configuration."""
    return get_config().generator


def get_validator_config() -> ValidatorConfig:
    """Get validator configuration."""
    return get_config().validator


def get_performance_config() -> PerformanceConfig:
    """Get performance configuration."""
    return get_config().performance


def get_logging_config() -> LoggingConfig:
    """Get logging configuration."""
    return get_config().logging