"""Exact fixed-shape launch exports from the unmodified original optimizer.

Original proposals supply one launch per input case, without an extrapolator.
This table preserves those choices; an unseen shape is rejected. Final tests
can use independent data on this explicitly fixed shape domain (spec 4.3).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class FrozenShapeLaunchRule:
    table: dict
    kind: str = "fixed_shape_table"
    version: int = 1

    def __post_init__(self):
        if self.kind != "fixed_shape_table" or self.version != 1 or not self.table:
            raise ValueError("Unsupported or empty frozen launch table")
        for key,launch in self.table.items():
            if not isinstance(key,str) or not key.isdigit() or int(key) < 1:
                raise ValueError("Frozen launch keys must be positive shape integers")
            _fixed_rule(launch)

    def apply(self,metadata,properties=None):
        n = metadata.get("n")
        if type(n) is not int or str(n) not in self.table:
            raise ValueError("Shape is outside the original optimizer's frozen launch table")
        return _fixed_rule(self.table[str(n)]).apply(metadata,properties)


def _fixed_rule(launch):
    from .evaluation import LaunchRule
    allowed = {"grid","block","shared_memory_bytes"}
    if not isinstance(launch,dict) or set(launch) != allowed:
        raise ValueError("A frozen shape must contain exactly grid/block/shared_memory_bytes")
    return LaunchRule(kind="fixed",**launch)


def parse_launch_rule(data):
    if data.get("kind") == "fixed_shape_table":
        return FrozenShapeLaunchRule(**data)
    from .evaluation import LaunchRule
    return LaunchRule(**data)
