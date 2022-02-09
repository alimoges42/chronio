from dataclasses import dataclass


@dataclass
class IndividualMetadata:
    metadata: dict

    def __post_init__(self):
        for key, val in self.metadata.items():
            setattr(self, key, val)


