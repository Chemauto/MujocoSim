"""common —— 全局通用层:配置加载、数据分类模型、数据总线/导出。

分层约束(整个仓库遵守,见 docs/architecture.md):
  common 不依赖任何其他内部包;
  sensors / simulation 只依赖 common;
  rl_control 依赖 common(经组合根注入 sensors 实例,不直接 import sensors);
  main / dashboard 是组合根,允许依赖一切。
"""
from .config import (ASSET_ROBOTS_DIR, DEFAULT_CONFIG, PROJECT_ROOT,
                     CameraCfg, ControllerCfg, DataCfg, ObjectCfg, SimConfig,
                     YoloCfg, load_config)
from .data_bus import DataBus, DataExporter, JsonlRecorder
from .data_model import EnvironmentState, RobotState, SensorData

__all__ = [
    "PROJECT_ROOT", "ASSET_ROBOTS_DIR", "DEFAULT_CONFIG",
    "SimConfig", "ObjectCfg", "CameraCfg", "YoloCfg", "ControllerCfg",
    "DataCfg", "load_config",
    "RobotState", "EnvironmentState", "SensorData",
    "DataBus", "DataExporter", "JsonlRecorder",
]
