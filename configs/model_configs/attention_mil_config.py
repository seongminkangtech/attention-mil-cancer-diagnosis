"""
Attention MIL 모델 설정

의료 영상 분류를 위한 Attention MIL 모델의 설정을 정의합니다.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import os


@dataclass
class FeatureExtractorConfig:
    """특징 추출기 설정"""
    model_name: str = "efficientnet_b2"
    pretrained: bool = True
    freeze_backbone: bool = False
    output_dim: int = 1408  # EfficientNet-B2 특징 차원
    use_global_pool: bool = True
    
    def __post_init__(self):
        """초기화 후 검증"""
        valid_models = [
            "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", 
            "efficientnet_b3", "efficientnet_b4", "efficientnet_b5",
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
            "densenet121", "densenet169", "densenet201"
        ]
        if self.model_name not in valid_models:
            raise ValueError(f"지원하지 않는 모델: {self.model_name}")


@dataclass
class AttentionConfig:
    """어텐션 메커니즘 설정"""
    hidden_dim: int = 128
    num_heads: int = 8
    dropout_rate: float = 0.2
    attention_type: str = "self_attention"  # "self_attention", "cross_attention"
    use_layer_norm: bool = True
    use_residual: bool = True
    
    def __post_init__(self):
        """초기화 후 검증"""
        if self.dropout_rate < 0 or self.dropout_rate > 1:
            raise ValueError(f"드롭아웃 비율은 0-1 사이여야 합니다: {self.dropout_rate}")
        if self.hidden_dim <= 0:
            raise ValueError(f"은닉 차원은 양수여야 합니다: {self.hidden_dim}")


@dataclass
class DataConfig:
    """데이터 처리 설정"""
    image_count: int = 50
    img_size: int = 256
    batch_size: int = 4
    train_ratio: float = 0.8
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    
    def __post_init__(self):
        """초기화 후 검증"""
        if self.train_ratio <= 0 or self.train_ratio >= 1:
            raise ValueError(f"훈련 비율은 0-1 사이여야 합니다: {self.train_ratio}")
        if self.batch_size <= 0:
            raise ValueError(f"배치 크기는 양수여야 합니다: {self.batch_size}")


@dataclass
class TrainingConfig:
    """학습 설정"""
    epochs: int = 50
    learning_rate: float = 2e-4
    optimizer: str = "adam"  # "adam", "sgd", "adamw"
    scheduler: str = "step"  # "step", "cosine", "plateau"
    step_size: int = 10
    gamma: float = 0.1
    weight_decay: float = 1e-4
    gradient_clip: Optional[float] = 1.0
    early_stopping_patience: int = 10
    save_best_only: bool = True
    
    def __post_init__(self):
        """초기화 후 검증"""
        if self.learning_rate <= 0:
            raise ValueError(f"학습률은 양수여야 합니다: {self.learning_rate}")
        if self.epochs <= 0:
            raise ValueError(f"에포크 수는 양수여야 합니다: {self.epochs}")


@dataclass
class PathsConfig:
    """경로 설정"""
    train_csv: str = "data/train.csv"
    test_csv: str = "data/test.csv"
    label_csv: str = "data/label_data.csv"
    frame_path: str = "data/frame/"
    model_save_path: str = "models/attention_mil/"
    checkpoint_path: str = "checkpoints/"
    log_path: str = "logs/"
    
    def __post_init__(self):
        """경로 생성"""
        Path(self.model_save_path).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)
        Path(self.log_path).mkdir(parents=True, exist_ok=True)


@dataclass
class HardwareConfig:
    """하드웨어 설정"""
    device: str = "auto"  # "auto", "cuda", "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = False
    deterministic: bool = False
    
    def __post_init__(self):
        """디바이스 자동 설정"""
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class LoggingConfig:
    """로깅 설정"""
    log_interval: int = 10
    save_interval: int = 5
    tensorboard: bool = True
    mlflow: bool = True
    wandb: bool = False
    log_gradients: bool = False
    log_learning_rate: bool = True
    
    def __post_init__(self):
        """환경 변수에서 설정 로드"""
        self.tensorboard = os.getenv("TENSORBOARD", str(self.tensorboard)).lower() == "true"
        self.mlflow = os.getenv("MLFLOW", str(self.mlflow)).lower() == "true"
        self.wandb = os.getenv("WANDB", str(self.wandb)).lower() == "true"


@dataclass
class EvaluationConfig:
    """평가 설정"""
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1", "auc"])
    save_predictions: bool = True
    confusion_matrix: bool = True
    roc_curve: bool = True
    pr_curve: bool = True
    threshold: float = 0.5
    
    def __post_init__(self):
        """초기화 후 검증"""
        valid_metrics = ["accuracy", "precision", "recall", "f1", "auc", "specificity", "sensitivity"]
        for metric in self.metrics:
            if metric not in valid_metrics:
                raise ValueError(f"지원하지 않는 메트릭: {metric}")


@dataclass
class DeploymentConfig:
    """배포 설정"""
    model_type: str = "onnx"  # "onnx", "torchscript", "tensorrt"
    onnx_model_path: str = "models/attention_mil.onnx"
    providers: List[str] = field(default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    optimization_level: int = 1
    enable_profiling: bool = False
    dynamic_axes: bool = True
    opset_version: int = 11
    
    def __post_init__(self):
        """초기화 후 검증"""
        valid_model_types = ["onnx", "torchscript", "tensorrt"]
        if self.model_type not in valid_model_types:
            raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")
        
        if self.optimization_level < 0 or self.optimization_level > 3:
            raise ValueError(f"최적화 레벨은 0-3 사이여야 합니다: {self.optimization_level}")


@dataclass
class AttentionMILConfig:
    """Attention MIL 모델 통합 설정"""
    name: str = "attention_mil"
    num_classes: int = 3
    
    # 하위 설정들
    feature_extractor: FeatureExtractorConfig = None
    attention: AttentionConfig = None
    data: DataConfig = None
    training: TrainingConfig = None
    paths: PathsConfig = None
    hardware: HardwareConfig = None
    logging: LoggingConfig = None
    evaluation: EvaluationConfig = None
    deployment: DeploymentConfig = None
    
    def __post_init__(self):
        """기본값 설정"""
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractorConfig()
        if self.attention is None:
            self.attention = AttentionConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.paths is None:
            self.paths = PathsConfig()
        if self.hardware is None:
            self.hardware = HardwareConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        if self.deployment is None:
            self.deployment = DeploymentConfig()
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "AttentionMILConfig":
        """YAML 파일에서 설정 로드 (하위 호환성)"""
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # YAML 설정을 dataclass로 변환
        config = cls()
        
        # 모델 설정
        if 'model' in config_dict:
            model_config = config_dict['model']
            config.num_classes = model_config.get('num_classes', config.num_classes)
            if 'feature_extractor' in model_config:
                fe_config = model_config['feature_extractor']
                config.feature_extractor.model_name = fe_config.get('model_name', config.feature_extractor.model_name)
                config.feature_extractor.pretrained = fe_config.get('pretrained', config.feature_extractor.pretrained)
            if 'attention' in model_config:
                att_config = model_config['attention']
                config.attention.hidden_dim = att_config.get('hidden_dim', config.attention.hidden_dim)
                config.attention.dropout_rate = att_config.get('dropout_rate', config.attention.dropout_rate)
        
        # 데이터 설정
        if 'data' in config_dict:
            data_config = config_dict['data']
            config.data.image_count = data_config.get('image_count', config.data.image_count)
            config.data.img_size = data_config.get('img_size', config.data.img_size)
            config.data.batch_size = data_config.get('batch_size', config.data.batch_size)
            config.data.train_ratio = data_config.get('train_ratio', config.data.train_ratio)
            config.data.shuffle = data_config.get('shuffle', config.data.shuffle)
        
        # 학습 설정
        if 'training' in config_dict:
            training_config = config_dict['training']
            config.training.epochs = training_config.get('epochs', config.training.epochs)
            config.training.learning_rate = training_config.get('learning_rate', config.training.learning_rate)
            config.training.optimizer = training_config.get('optimizer', config.training.optimizer)
            config.training.scheduler = training_config.get('scheduler', config.training.scheduler)
            config.training.step_size = training_config.get('step_size', config.training.step_size)
            config.training.gamma = training_config.get('gamma', config.training.gamma)
        
        # 경로 설정
        if 'paths' in config_dict:
            paths_config = config_dict['paths']
            config.paths.train_csv = paths_config.get('train_csv', config.paths.train_csv)
            config.paths.test_csv = paths_config.get('test_csv', config.paths.test_csv)
            config.paths.label_csv = paths_config.get('label_csv', config.paths.label_csv)
            config.paths.frame_path = paths_config.get('frame_path', config.paths.frame_path)
            config.paths.model_save_path = paths_config.get('model_save_path', config.paths.model_save_path)
        
        # 하드웨어 설정
        if 'hardware' in config_dict:
            hardware_config = config_dict['hardware']
            config.hardware.device = hardware_config.get('device', config.hardware.device)
            config.hardware.num_workers = hardware_config.get('num_workers', config.hardware.num_workers)
        
        # 로깅 설정
        if 'logging' in config_dict:
            logging_config = config_dict['logging']
            config.logging.log_interval = logging_config.get('log_interval', config.logging.log_interval)
            config.logging.save_interval = logging_config.get('save_interval', config.logging.save_interval)
            config.logging.tensorboard = logging_config.get('tensorboard', config.logging.tensorboard)
            config.logging.mlflow = logging_config.get('mlflow', config.logging.mlflow)
        
        # 평가 설정
        if 'evaluation' in config_dict:
            evaluation_config = config_dict['evaluation']
            config.evaluation.metrics = evaluation_config.get('metrics', config.evaluation.metrics)
            config.evaluation.save_predictions = evaluation_config.get('save_predictions', config.evaluation.save_predictions)
            config.evaluation.confusion_matrix = evaluation_config.get('confusion_matrix', config.evaluation.confusion_matrix)
        
        # 배포 설정
        if 'deployment' in config_dict:
            deployment_config = config_dict['deployment']
            config.deployment.model_type = deployment_config.get('model_type', config.deployment.model_type)
            config.deployment.onnx_model_path = deployment_config.get('onnx_model_path', config.deployment.onnx_model_path)
            config.deployment.providers = deployment_config.get('providers', config.deployment.providers)
            config.deployment.optimization_level = deployment_config.get('optimization_level', config.deployment.optimization_level)
            config.deployment.enable_profiling = deployment_config.get('enable_profiling', config.deployment.enable_profiling)
        
        return config
