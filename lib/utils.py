"""Utility files for project."""

import yaml
import abc


class AbstractYAMLMeta(yaml.YAMLObjectMetaclass, abc.ABCMeta):
    """Metaclass used to fix conflicts in multiple inheritance."""

    def __init__(cls, name, bases, kwds):
        """Initialize class and set parameters."""
        super().__init__(name, bases, kwds)
        cls.yaml_tag = f"!{cls.__name__}"
        cls.yaml_loader.add_constructor(f"!{cls.__name__}", cls.from_yaml)
        cls.yaml_dumper.add_representer(cls, cls.to_yaml)
