"""
환경별 설정 관리

개발, 스테이징, 프로덕션 환경별 설정을 정의합니다.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
import os


class Environment(Enum):
    """환경 타입"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


@dataclass
class EnvironmentConfig:
    """환경별 설정"""
    env: Environment = Environment.DEVELOPMENT
    debug: bool = True
    log_level: str = "DEBUG"
    reload: bool = True
    workers: int = 1
    host: str = "0.0.0.0"
    port: int = 8000
    
    # 성능 설정
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    batch_size: int = 4
    
    # 모니터링 설정
    metrics_enabled: bool = True
    health_check_interval: int = 30
    readiness_probe_timeout: int = 5
    liveness_probe_timeout: int = 5
    
    def __post_init__(self):
        """환경 변수에서 설정 로드"""
        env_str = os.getenv("ENVIRONMENT", "development").lower()
        self.env = Environment(env_str)
        
        # 환경별 기본값 설정
        if self.env == Environment.PRODUCTION:
            self.debug = False
            self.log_level = "INFO"
            self.reload = False
            self.workers = 4
            self.max_concurrent_requests = 100
        elif self.env == Environment.STAGING:
            self.debug = False
            self.log_level = "INFO"
            self.reload = False
            self.workers = 2
        elif self.env == Environment.TEST:
            self.debug = True
            self.log_level = "DEBUG"
            self.reload = True
            self.workers = 1
        
        # 환경 변수에서 개별 설정 로드
        self.debug = os.getenv("DEBUG", str(self.debug)).lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        self.reload = os.getenv("API_RELOAD", str(self.reload)).lower() == "true"
        self.workers = int(os.getenv("API_WORKERS", self.workers))
        self.host = os.getenv("API_HOST", self.host)
        self.port = int(os.getenv("API_PORT", self.port))
        self.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS", self.max_concurrent_requests))
        self.request_timeout = int(os.getenv("INFERENCE_TIMEOUT", self.request_timeout))
        self.batch_size = int(os.getenv("BATCH_SIZE", self.batch_size))
        self.metrics_enabled = os.getenv("METRICS_ENABLED", str(self.metrics_enabled)).lower() == "true"
    
    @property
    def is_production(self) -> bool:
        """프로덕션 환경 여부"""
        return self.env == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """개발 환경 여부"""
        return self.env == Environment.DEVELOPMENT
    
    @property
    def is_staging(self) -> bool:
        """스테이징 환경 여부"""
        return self.env == Environment.STAGING
    
    @property
    def is_test(self) -> bool:
        """테스트 환경 여부"""
        return self.env == Environment.TEST
    
    def get_environment_specific_config(self) -> Dict[str, Any]:
        """환경별 특화 설정 반환"""
        if self.is_production:
            return {
                "cors_origins": ["https://yourdomain.com"],
                "allowed_hosts": ["yourdomain.com"],
                "log_level": "WARNING",
                "enable_profiling": False
            }
        elif self.is_staging:
            return {
                "cors_origins": ["https://staging.yourdomain.com"],
                "allowed_hosts": ["staging.yourdomain.com"],
                "log_level": "INFO",
                "enable_profiling": True
            }
        else:  # development, test
            return {
                "cors_origins": ["*"],
                "allowed_hosts": ["localhost", "127.0.0.1"],
                "log_level": "DEBUG",
                "enable_profiling": True
            }
