"""
기본 설정 클래스들

데이터베이스, 로깅, MLflow, 보안 등 공통 설정을 정의합니다.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import os


@dataclass
class DatabaseConfig:
    """데이터베이스 연결 설정"""
    host: str = "localhost"
    port: int = 5432
    database: str = "medical_ai"
    username: str = "user"
    password: str = "password"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False
    
    @property
    def connection_string(self) -> str:
        """데이터베이스 연결 문자열 생성"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class LoggingConfig:
    """로깅 설정"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[Path] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True
    file_output: bool = False
    
    def __post_init__(self):
        """초기화 후 검증"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            raise ValueError(f"유효하지 않은 로그 레벨: {self.level}")


@dataclass
class MLflowConfig:
    """MLflow 설정"""
    tracking_uri: str = "http://localhost:5000"
    registry_uri: str = "http://localhost:5000"
    experiment_name: str = "attention-mil-cancer-diagnosis"
    model_registry_name: str = "attention-mil-models"
    tracking_username: Optional[str] = None
    tracking_password: Optional[str] = None
    artifact_location: Optional[str] = None
    enable_autolog: bool = True
    
    def __post_init__(self):
        """환경 변수에서 설정 로드"""
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", self.tracking_uri)
        self.registry_uri = os.getenv("MLFLOW_REGISTRY_URI", self.registry_uri)
        self.experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", self.experiment_name)
        self.tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME", self.tracking_username)
        self.tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD", self.tracking_password)


@dataclass
class SecurityConfig:
    """보안 설정"""
    secret_key: str = "your-secret-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    allowed_hosts: List[str] = field(default_factory=lambda: ["localhost", "127.0.0.1"])
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1시간
    
    def __post_init__(self):
        """환경 변수에서 설정 로드"""
        self.secret_key = os.getenv("SECRET_KEY", self.secret_key)
        
        # 프로덕션 환경에서는 보안 강화
        if os.getenv("ENVIRONMENT") == "production":
            if self.cors_origins == ["*"]:
                self.cors_origins = ["https://yourdomain.com"]
            if self.secret_key == "your-secret-key-here":
                raise ValueError("프로덕션 환경에서는 SECRET_KEY를 설정해야 합니다.")
