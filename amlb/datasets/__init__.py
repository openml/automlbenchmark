from enum import Enum, auto

from .openml import Openml


class DataSourceType(Enum):
    openml_task = auto()
    openml_dataset = auto()
    local = auto()


class DataLoader:

    def __init__(self, config):
        self.openml_loader = Openml(api_key=config.openml.apikey, cache_dir=config.input_dir)

    def load(self, source: DataSourceType, *args, **kwargs):
        if source == DataSourceType.openml_task:
            self.openml_loader.load(*args, **kwargs)
        elif source == DataSourceType.local:
             pass
        else:
            raise NotImplementedError(f"data source {source} is not supported yet")


__all__ = (DataLoader, DataSourceType)