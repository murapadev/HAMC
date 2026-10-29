"""Tunable constants read by the generators and validators."""

from dataclasses import dataclass


@dataclass
class GeneratorConfig:
    local_river_water_probability: float = 0.8
    local_river_grass_probability: float = 0.2
    local_desert_sand_probability: float = 0.9
    local_desert_grass_probability: float = 0.1
    local_water_probability_decay: float = 0.9
    local_min_water_probability: float = 0.0


@dataclass
class ValidatorConfig:
    min_water_percentage: float = 0.1


_generator_config = GeneratorConfig()
_validator_config = ValidatorConfig()


def get_generator_config() -> GeneratorConfig:
    return _generator_config


def get_validator_config() -> ValidatorConfig:
    return _validator_config
