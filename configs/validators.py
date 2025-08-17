"""
설정 검증 유틸리티

설정의 유효성을 검증하는 함수들을 포함합니다.
"""

import os
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def validate_config(config: Any) -> bool:
    """
    설정 객체의 유효성 검증

    Args:
        config: 검증할 설정 객체

    Returns:
        bool: 검증 성공 여부

    Raises:
        ValueError: 설정이 유효하지 않은 경우
    """
    if not is_dataclass(config):
        raise ValueError("설정은 dataclass여야 합니다.")

    # 모든 필드 검증
    for field in fields(config):
        value = getattr(config, field.name)
        if not validate_field(field.name, value, field.type):
            raise ValueError(f"필드 '{field.name}'의 값이 유효하지 않습니다: {value}")

    return True


def validate_field(field_name: str, value: Any, field_type: Any) -> bool:
    """
    개별 필드 값 검증

    Args:
        field_name: 필드 이름
        value: 검증할 값
        field_type: 필드 타입

    Returns:
        bool: 검증 성공 여부
    """
    # None 값 검증
    if value is None:
        return True  # None은 허용

    # 기본 타입 검증
    if field_type == str:
        return isinstance(value, str) and len(value) > 0
    elif field_type == int:
        return isinstance(value, int) and value >= 0
    elif field_type == float:
        return isinstance(value, (int, float)) and value >= 0
    elif field_type == bool:
        return isinstance(value, bool)
    elif field_type == Path:
        return isinstance(value, (str, Path))

    # 리스트 검증
    if hasattr(field_type, "__origin__") and field_type.__origin__ == list:
        if not isinstance(value, list):
            return False
        if len(value) == 0:
            return True  # 빈 리스트는 허용

        # 리스트 내부 타입 검증
        item_type = field_type.__args__[0]
        return all(
            validate_field(f"{field_name}[{i}]", item, item_type)
            for i, item in enumerate(value)
        )

    # 딕셔너리 검증
    if hasattr(field_type, "__origin__") and field_type.__origin__ == dict:
        if not isinstance(value, dict):
            return False
        return True  # 딕셔너리는 기본적으로 허용

    # dataclass 검증
    if is_dataclass(field_type):
        if not isinstance(value, field_type):
            return False
        return validate_config(value)

    # 기타 타입은 기본적으로 허용
    return True


def validate_paths(config: Any) -> bool:
    """
    경로 설정 검증

    Args:
        config: 검증할 설정 객체

    Returns:
        bool: 검증 성공 여부
    """
    if not is_dataclass(config):
        return False

    # 경로 관련 필드 찾기
    path_fields = []
    for field in fields(config):
        if "path" in field.name.lower() or field.name in [
            "train_csv",
            "test_csv",
            "label_csv",
        ]:
            path_fields.append(field.name)

    # 경로 검증
    for field_name in path_fields:
        value = getattr(config, field_name, None)
        if value and isinstance(value, (str, Path)):
            path = Path(value)
            # 파일 경로인 경우 부모 디렉토리 존재 확인
            if path.suffix:  # 파일인 경우
                parent_dir = path.parent
                if not parent_dir.exists():
                    print(f"⚠️ 경고: 디렉토리가 존재하지 않습니다: {parent_dir}")
            # 디렉토리인 경우 존재 확인
            elif not path.exists():
                print(f"⚠️ 경고: 디렉토리가 존재하지 않습니다: {path}")

    return True


def validate_environment_config(config: Any) -> bool:
    """
    환경별 설정 검증

    Args:
        config: 검증할 설정 객체

    Returns:
        bool: 검증 성공 여부
    """
    if not is_dataclass(config):
        return False

    # 환경 변수 확인
    env = os.getenv("ENVIRONMENT", "development").lower()

    # 프로덕션 환경 검증
    if env == "production":
        # 보안 설정 검증
        if hasattr(config, "security"):
            security = config.security
            if hasattr(security, "secret_key"):
                if security.secret_key == "your-secret-key-here":
                    print("❌ 프로덕션 환경에서는 SECRET_KEY를 설정해야 합니다.")
                    return False

        # CORS 설정 검증
        if hasattr(config, "fastapi") and hasattr(config.fastapi, "cors"):
            cors = config.fastapi.cors
            if hasattr(cors, "allow_origins"):
                if "*" in cors.allow_origins:
                    print("❌ 프로덕션 환경에서는 CORS를 '*'로 설정할 수 없습니다.")
                    return False

    return True


def validate_model_config(config: Any) -> bool:
    """
    모델 설정 검증

    Args:
        config: 검증할 설정 객체

    Returns:
        bool: 검증 성공 여부
    """
    if not is_dataclass(config):
        return False

    # 모델 관련 필드 검증
    if hasattr(config, "attention_mil"):
        model_config = config.attention_mil

        # 클래스 수 검증
        if hasattr(model_config, "num_classes"):
            if model_config.num_classes <= 0:
                print("❌ 클래스 수는 양수여야 합니다.")
                return False

        # 특징 추출기 검증
        if hasattr(model_config, "feature_extractor"):
            fe_config = model_config.feature_extractor
            if hasattr(fe_config, "output_dim"):
                if fe_config.output_dim <= 0:
                    print("❌ 특징 차원은 양수여야 합니다.")
                    return False

        # 어텐션 설정 검증
        if hasattr(model_config, "attention"):
            att_config = model_config.attention
            if hasattr(att_config, "hidden_dim"):
                if att_config.hidden_dim <= 0:
                    print("❌ 어텐션 은닉 차원은 양수여야 합니다.")
                    return False
            if hasattr(att_config, "dropout_rate"):
                if not (0 <= att_config.dropout_rate <= 1):
                    print("❌ 드롭아웃 비율은 0-1 사이여야 합니다.")
                    return False

    return True


def validate_api_config(config: Any) -> bool:
    """
    API 설정 검증

    Args:
        config: 검증할 설정 객체

    Returns:
        bool: 검증 성공 여부
    """
    if not is_dataclass(config):
        return False

    # API 관련 필드 검증
    if hasattr(config, "fastapi"):
        api_config = config.fastapi

        # 포트 검증
        if hasattr(api_config, "port"):
            if not (1024 <= api_config.port <= 65535):
                print("❌ 포트는 1024-65535 사이여야 합니다.")
                return False

        # 워커 수 검증
        if hasattr(api_config, "workers"):
            if api_config.workers <= 0:
                print("❌ 워커 수는 양수여야 합니다.")
                return False

        # 로그 레벨 검증
        if hasattr(api_config, "log_level"):
            valid_levels = ["debug", "info", "warning", "error", "critical"]
            if api_config.log_level.lower() not in valid_levels:
                print(f"❌ 유효하지 않은 로그 레벨: {api_config.log_level}")
                return False

    return True


def comprehensive_validation(config: Any) -> Dict[str, bool]:
    """
    종합적인 설정 검증

    Args:
        config: 검증할 설정 객체

    Returns:
        Dict[str, bool]: 각 검증 단계별 결과
    """
    results = {}

    try:
        # 기본 검증
        results["basic"] = validate_config(config)
    except Exception as e:
        results["basic"] = False
        print(f"❌ 기본 검증 실패: {e}")

    try:
        # 경로 검증
        results["paths"] = validate_paths(config)
    except Exception as e:
        results["paths"] = False
        print(f"❌ 경로 검증 실패: {e}")

    try:
        # 환경별 검증
        results["environment"] = validate_environment_config(config)
    except Exception as e:
        results["environment"] = False
        print(f"❌ 환경별 검증 실패: {e}")

    try:
        # 모델 검증
        results["model"] = validate_model_config(config)
    except Exception as e:
        results["model"] = False
        print(f"❌ 모델 검증 실패: {e}")

    try:
        # API 검증
        results["api"] = validate_api_config(config)
    except Exception as e:
        results["api"] = False
        print(f"❌ API 검증 실패: {e}")

    # 전체 결과
    results["overall"] = all(results.values())

    return results
