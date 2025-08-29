#!/usr/bin/env python3
"""
HAMC Configuration Demo Script
Demonstrates the advanced configuration system capabilities.
"""

import json
import time
from pathlib import Path
from hamc.config.advanced_config import (
    get_config, load_config, save_config, ConfigProfile,
    get_entropy_config, get_cache_config, get_generator_config,
    get_validator_config, get_performance_config, get_logging_config
)

def demo_configuration_loading():
    """Demonstrate different ways to load configuration."""
    print("🔧 Configuration Loading Demo")
    print("=" * 50)

    # Load default configuration
    print("1. Loading default configuration...")
    config = get_config()
    print(f"   Active profile: {config.active_profile.value}")
    print(f"   Default temperature: {config.entropy.default_temperature}")

    # Load production configuration
    print("\n2. Loading production configuration...")
    config_path = "hamc/config/profiles/production.json"
    if Path(config_path).exists():
        prod_config = load_config(config_path, ConfigProfile.PRODUCTION)
        print(f"   Production max workers: {prod_config.performance.max_workers}")
        print(f"   Production cache size: {prod_config.cache.compatibility_cache_max_size}")
    else:
        print("   Production config file not found")

    # Load testing configuration
    print("\n3. Loading testing configuration...")
    config_path = "hamc/config/profiles/testing.json"
    if Path(config_path).exists():
        test_config = load_config(config_path, ConfigProfile.TESTING)
        print(f"   Testing max workers: {test_config.performance.max_workers}")
        print(f"   Testing profiling: {test_config.performance.enable_profiling}")
    else:
        print("   Testing config file not found")

def demo_configuration_access():
    """Demonstrate how to access configuration values."""
    print("\n📖 Configuration Access Demo")
    print("=" * 50)

    # Access different configuration sections
    entropy = get_entropy_config()
    cache = get_cache_config()
    generator = get_generator_config()
    validator = get_validator_config()
    performance = get_performance_config()
    logging = get_logging_config()

    print("1. Entropy Configuration:")
    print(f"   Default temperature: {entropy.default_temperature}")
    print(f"   Learning rate: {entropy.default_learning_rate}")
    print(f"   Min success rate: {entropy.min_success_rate}")

    print("\n2. Cache Configuration:")
    print(f"   Compatibility cache size: {cache.compatibility_cache_max_size}")
    print(f"   Cache TTL: {cache.compatibility_cache_ttl}")
    print(f"   Min hit ratio: {cache.min_hit_ratio}")

    print("\n3. Generator Configuration:")
    print(f"   River water probability: {generator.local_river_water_probability}")
    print(f"   Road asphalt probability: {generator.local_road_asphalt_probability}")
    print(f"   Desert sand probability: {generator.local_desert_sand_probability}")

    print("\n4. Validator Configuration:")
    print(f"   Min water percentage: {validator.min_water_percentage}")
    print(f"   Strict compatibility: {validator.strict_compatibility}")

    print("\n5. Performance Configuration:")
    print(f"   Max workers: {performance.max_workers}")
    print(f"   Enable parallel: {performance.enable_parallel_processing}")
    print(f"   Enable profiling: {performance.enable_profiling}")

    print("\n6. Logging Configuration:")
    print(f"   Default level: {logging.default_level}")
    print(f"   Log directory: {logging.log_dir}")
    print(f"   Max log size: {logging.max_log_size_mb}MB")

def demo_runtime_modification():
    """Demonstrate runtime configuration modification."""
    print("\n⚙️ Runtime Configuration Modification Demo")
    print("=" * 50)

    config = get_config()
    original_temp = config.entropy.default_temperature
    original_workers = config.performance.max_workers

    print(f"Original temperature: {original_temp}")
    print(f"Original max workers: {original_workers}")

    # Modify configuration at runtime
    print("\nModifying configuration...")
    config.entropy.default_temperature = 1.5
    config.performance.max_workers = 8

    print(f"Modified temperature: {config.entropy.default_temperature}")
    print(f"Modified max workers: {config.performance.max_workers}")

    # Demonstrate dot notation access
    print("\nUsing dot notation access:")
    print(f"config.entropy.min_temperature: {config.entropy.min_temperature}")
    print(f"config.cache.pattern_cache_max_size: {config.cache.pattern_cache_max_size}")

def demo_configuration_persistence():
    """Demonstrate saving and loading configuration."""
    print("\n💾 Configuration Persistence Demo")
    print("=" * 50)

    # Create a custom configuration
    config = get_config()
    config.entropy.default_temperature = 2.0
    config.performance.max_workers = 12
    config.cache.compatibility_cache_max_size = 8192

    # Save to a temporary file
    temp_config_path = "hamc/config/profiles/custom_demo.json"
    print(f"Saving custom configuration to: {temp_config_path}")
    save_config(temp_config_path, config)

    # Load it back
    print("Loading custom configuration...")
    loaded_config = load_config(temp_config_path)
    print(f"Loaded temperature: {loaded_config.entropy.default_temperature}")
    print(f"Loaded max workers: {loaded_config.performance.max_workers}")
    print(f"Loaded cache size: {loaded_config.cache.compatibility_cache_max_size}")

    # Clean up
    Path(temp_config_path).unlink(missing_ok=True)
    print("Cleaned up temporary configuration file")

def demo_performance_impact():
    """Demonstrate the performance impact of configuration access."""
    print("\n⚡ Performance Impact Demo")
    print("=" * 50)

    config = get_config()

    # Measure configuration access time
    iterations = 10000

    start_time = time.time()
    for _ in range(iterations):
        temp = config.entropy.default_temperature
        workers = config.performance.max_workers
        cache_size = config.cache.compatibility_cache_max_size
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = (total_time / iterations) * 1000  # Convert to milliseconds

    print(f"Configuration access performance:")
    print(f"   Iterations: {iterations}")
    print(f"   Total time: {total_time:.4f} seconds")
    print(f"   Average time per access: {avg_time:.4f} milliseconds")
    print(f"   Accesses per second: {iterations / total_time:.0f}")

def main():
    """Main demo function."""
    print("🚀 HAMC Advanced Configuration System Demo")
    print("=" * 60)
    print("This demo showcases the revolutionary configuration system")
    print("that eliminates hardcoded values and enables dynamic behavior.\n")

    try:
        demo_configuration_loading()
        demo_configuration_access()
        demo_runtime_modification()
        demo_configuration_persistence()
        demo_performance_impact()

        print("\n✅ Configuration Demo Completed Successfully!")
        print("\nKey Benefits:")
        print("• Zero hardcoded values in the codebase")
        print("• Profile-based configuration for different environments")
        print("• Runtime configuration modification")
        print("• Type-safe configuration access")
        print("• Minimal performance impact")
        print("• Easy extensibility for new parameters")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()