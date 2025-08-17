"""
FastAPI 서버 설정

의료 AI 추론 서비스를 위한 FastAPI 설정을 정의합니다.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CORSConfig:
    """CORS 설정"""

    allow_origins: List[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = True
    allow_methods: List[str] = field(default_factory=lambda: ["*"])
    allow_headers: List[str] = field(default_factory=lambda: ["*"])
    expose_headers: List[str] = field(default_factory=lambda: [])
    max_age: int = 600

    def __post_init__(self):
        """환경별 CORS 설정 조정"""
        env = os.getenv("ENVIRONMENT", "development").lower()
        if env == "production":
            # 프로덕션에서는 특정 도메인만 허용
            if self.allow_origins == ["*"]:
                self.allow_origins = ["https://yourdomain.com"]
        elif env == "staging":
            # 스테이징에서는 스테이징 도메인 허용
            if self.allow_origins == ["*"]:
                self.allow_origins = ["https://staging.yourdomain.com"]


@dataclass
class MiddlewareConfig:
    """미들웨어 설정"""

    enable_cors: bool = True
    enable_logging: bool = True
    enable_auth: bool = False
    enable_rate_limit: bool = True
    enable_compression: bool = True
    enable_trusted_host: bool = True

    # 로깅 미들웨어 설정
    log_requests: bool = True
    log_responses: bool = True
    log_exceptions: bool = True

    # 인증 미들웨어 설정
    jwt_secret: str = "your-jwt-secret"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30

    # 속도 제한 설정
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1시간

    def __post_init__(self):
        """환경 변수에서 설정 로드"""
        self.enable_auth = (
            os.getenv("ENABLE_AUTH", str(self.enable_auth)).lower() == "true"
        )
        self.enable_rate_limit = (
            os.getenv("ENABLE_RATE_LIMIT", str(self.enable_rate_limit)).lower()
            == "true"
        )
        self.jwt_secret = os.getenv("JWT_SECRET", self.jwt_secret)


@dataclass
class FastAPIConfig:
    """FastAPI 서버 설정"""

    title: str = "의료 AI 추론 서비스"
    description: str = "의료 영상 분류를 위한 Attention MIL 모델 API"
    version: str = "1.0.0"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"

    # 서버 설정
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    workers: int = 1
    log_level: str = "info"

    # 성능 설정
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    max_request_size: int = 10 * 1024 * 1024  # 10MB

    # 보안 설정
    enable_https: bool = False
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None

    # 미들웨어 설정
    cors: CORSConfig = None
    middleware: MiddlewareConfig = None

    def __post_init__(self):
        """기본값 설정 및 환경 변수 로드"""
        if self.cors is None:
            self.cors = CORSConfig()
        if self.middleware is None:
            self.middleware = MiddlewareConfig()

        # 환경 변수에서 설정 로드
        self.host = os.getenv("API_HOST", self.host)
        self.port = int(os.getenv("API_PORT", self.port))
        self.reload = os.getenv("API_RELOAD", str(self.reload)).lower() == "true"
        self.workers = int(os.getenv("API_WORKERS", self.workers))
        self.log_level = os.getenv("LOG_LEVEL", self.log_level).lower()
        self.max_concurrent_requests = int(
            os.getenv("MAX_CONCURRENT_REQUESTS", self.max_concurrent_requests)
        )
        self.request_timeout = int(os.getenv("INFERENCE_TIMEOUT", self.request_timeout))
        self.max_request_size = int(os.getenv("MAX_FILE_SIZE", self.max_request_size))

        # 환경별 설정 조정
        env = os.getenv("ENVIRONMENT", "development").lower()
        if env == "production":
            self.reload = False
            self.workers = max(4, self.workers)
            self.log_level = "warning"
            self.enable_https = True
        elif env == "staging":
            self.reload = False
            self.workers = max(2, self.workers)
            self.log_level = "info"

    def get_server_config(self) -> Dict[str, Any]:
        """서버 실행 설정 반환"""
        return {
            "host": self.host,
            "port": self.port,
            "reload": self.reload,
            "workers": self.workers,
            "log_level": self.log_level,
            "access_log": True,
        }

    def get_app_config(self) -> Dict[str, Any]:
        """FastAPI 앱 설정 반환"""
        return {
            "title": self.title,
            "description": self.description,
            "version": self.version,
            "docs_url": self.docs_url,
            "redoc_url": self.redoc_url,
            "openapi_url": self.openapi_url,
        }

    def get_cors_config(self) -> Dict[str, Any]:
        """CORS 설정 반환"""
        return {
            "allow_origins": self.cors.allow_origins,
            "allow_credentials": self.cors.allow_credentials,
            "allow_methods": self.cors.allow_methods,
            "allow_headers": self.cors.allow_headers,
            "expose_headers": self.cors.expose_headers,
            "max_age": self.cors.max_age,
        }
