"""
통합 애플리케이션 설정

모든 설정 클래스를 통합하여 관리하는 메인 설정 클래스입니다.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import os
import yaml

from .base_config import DatabaseConfig, LoggingConfig, MLflowConfig, SecurityConfig
from .environment_config import Environment, EnvironmentConfig
from .model_configs.attention_mil_config import AttentionMILConfig
from .api_configs.fastapi_config import FastAPIConfig


@dataclass
class AppConfig:
    """통합 애플리케이션 설정"""
    
    # 환경 설정
    environment: EnvironmentConfig = None
    
    # 기본 설정
    database: DatabaseConfig = None
    logging: LoggingConfig = None
    mlflow: MLflowConfig = None
    security: SecurityConfig = None
    
    # 모델 설정
    attention_mil: AttentionMILConfig = None
    
    # API 설정
    fastapi: FastAPIConfig = None
    
    def __post_init__(self):
        """기본값 설정"""
        if self.environment is None:
            self.environment = EnvironmentConfig()
        if self.database is None:
            self.database = DatabaseConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.mlflow is None:
            self.mlflow = MLflowConfig()
        if self.security is None:
            self.security = SecurityConfig()
        if self.attention_mil is None:
            self.attention_mil = AttentionMILConfig()
        if self.fastapi is None:
            self.fastapi = FastAPIConfig()
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """환경 변수에서 설정 로드"""
        config = cls()
        
        # 환경별 설정 조정
        env = os.getenv("ENVIRONMENT", "development").lower()
        if env == "production":
            config.logging.level = "WARNING"
            config.fastapi.reload = False
            config.fastapi.workers = 4
            config.security.cors_origins = ["https://yourdomain.com"]
        elif env == "staging":
            config.logging.level = "INFO"
            config.fastapi.reload = False
            config.fastapi.workers = 2
            config.security.cors_origins = ["https://staging.yourdomain.com"]
        
        return config
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "AppConfig":
        """YAML 파일에서 설정 로드 (하위 호환성)"""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {yaml_path}")
        
        # 기본 설정 로드
        config = cls.from_env()
        
        # YAML 파일 로드
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        
        # Attention MIL 모델 설정 로드
        if yaml_config:
            config.attention_mil = AttentionMILConfig.from_yaml(yaml_path)
        
        return config
    
    @classmethod
    def from_config_dir(cls, config_dir: str = "configs") -> "AppConfig":
        """설정 디렉토리에서 모든 설정 로드"""
        config = cls.from_env()
        
        config_path = Path(config_dir)
        
        # 모델 설정 로드
        model_config_path = config_path / "model_configs" / "attention_mil.yaml"
        if model_config_path.exists():
            config.attention_mil = AttentionMILConfig.from_yaml(str(model_config_path))
        
        # API 설정 로드 (YAML 파일이 있다면)
        api_config_path = config_path / "api_configs" / "fastapi.yaml"
        if api_config_path.exists():
            with open(api_config_path, 'r', encoding='utf-8') as f:
                api_yaml = yaml.safe_load(f)
                # YAML 설정을 FastAPIConfig에 적용
                if api_yaml:
                    for key, value in api_yaml.items():
                        if hasattr(config.fastapi, key):
                            setattr(config.fastapi, key, value)
        
        return config
    
    def validate(self) -> bool:
        """설정 유효성 검증"""
        try:
            # 환경 설정 검증
            if not self.environment:
                raise ValueError("환경 설정이 누락되었습니다.")
            
            # 모델 설정 검증
            if not self.attention_mil:
                raise ValueError("Attention MIL 모델 설정이 누락되었습니다.")
            
            # API 설정 검증
            if not self.fastapi:
                raise ValueError("FastAPI 설정이 누락되었습니다.")
            
            # 경로 설정 검증
            if self.attention_mil.paths:
                required_paths = [
                    self.attention_mil.paths.train_csv,
                    self.attention_mil.paths.label_csv,
                    self.attention_mil.paths.frame_path
                ]
                for path in required_paths:
                    if not path:
                        raise ValueError(f"필수 경로가 누락되었습니다: {path}")
            
            return True
            
        except Exception as e:
            print(f"설정 검증 실패: {e}")
            return False
    
    def get_model_config(self) -> Dict[str, Any]:
        """모델 설정을 딕셔너리로 반환"""
        return {
            "num_classes": self.attention_mil.num_classes,
            "feature_extractor": {
                "model_name": self.attention_mil.feature_extractor.model_name,
                "pretrained": self.attention_mil.feature_extractor.pretrained,
                "output_dim": self.attention_mil.feature_extractor.output_dim
            },
            "attention": {
                "hidden_dim": self.attention_mil.attention.hidden_dim,
                "dropout_rate": self.attention_mil.attention.dropout_rate
            }
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """학습 설정을 딕셔너리로 반환"""
        return {
            "epochs": self.attention_mil.training.epochs,
            "learning_rate": self.attention_mil.training.learning_rate,
            "batch_size": self.attention_mil.data.batch_size,
            "device": self.attention_mil.hardware.device
        }
    
    def get_api_config(self) -> Dict[str, Any]:
        """API 설정을 딕셔너리로 반환"""
        return {
            "host": self.fastapi.host,
            "port": self.fastapi.port,
            "reload": self.fastapi.reload,
            "workers": self.fastapi.workers,
            "log_level": self.fastapi.log_level
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """전체 설정을 딕셔너리로 변환"""
        return {
            "environment": {
                "env": self.environment.env.value,
                "debug": self.environment.debug,
                "log_level": self.environment.log_level
            },
            "model": self.get_model_config(),
            "training": self.get_training_config(),
            "api": self.get_api_config(),
            "mlflow": {
                "tracking_uri": self.mlflow.tracking_uri,
                "experiment_name": self.mlflow.experiment_name
            }
        }
    
    def print_summary(self):
        """설정 요약 출력"""
        print("=" * 60)
        print("🔧 애플리케이션 설정 요약")
        print("=" * 60)
        
        print(f"🌍 환경: {self.environment.env.value}")
        print(f"🐛 디버그 모드: {self.environment.debug}")
        print(f"📝 로그 레벨: {self.environment.log_level}")
        
        print(f"\n🤖 모델 설정:")
        print(f"   - 클래스 수: {self.attention_mil.num_classes}")
        print(f"   - 특징 추출기: {self.attention_mil.feature_extractor.model_name}")
        print(f"   - 어텐션 차원: {self.attention_mil.attention.hidden_dim}")
        
        print(f"\n🚀 API 설정:")
        print(f"   - 호스트: {self.fastapi.host}")
        print(f"   - 포트: {self.fastapi.port}")
        print(f"   - 워커 수: {self.fastapi.workers}")
        print(f"   - 자동 리로드: {self.fastapi.reload}")
        
        print(f"\n📊 MLflow 설정:")
        print(f"   - 추적 URI: {self.mlflow.tracking_uri}")
        print(f"   - 실험 이름: {self.mlflow.experiment_name}")
        
        print("=" * 60)
