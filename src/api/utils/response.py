"""
API 응답 포맷 유틸리티

일관된 API 응답 형식을 제공합니다.
"""

from datetime import datetime
from typing import Any, Dict, Optional


def create_response(
    success: bool,
    message: str,
    data: Optional[Any] = None,
    error_code: Optional[str] = None,
) -> Dict[str, Any]:
    """
    표준화된 API 응답 생성

    Args:
        success (bool): 요청 성공 여부
        message (str): 응답 메시지
        data (Optional[Any]): 응답 데이터
        error_code (Optional[str]): 오류 코드

    Returns:
        Dict[str, Any]: 표준화된 응답 딕셔너리
    """
    response = {
        "success": success,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "service": "의료 AI 추론 서비스",
    }

    if data is not None:
        response["data"] = data

    if error_code is not None:
        response["error_code"] = error_code

    return response


def create_error_response(
    message: str, error_code: str, details: Optional[Any] = None
) -> Dict[str, Any]:
    """
    오류 응답 생성

    Args:
        message (str): 오류 메시지
        error_code (str): 오류 코드
        details (Optional[Any]): 상세 오류 정보

    Returns:
        Dict[str, Any]: 오류 응답 딕셔너리
    """
    return create_response(
        success=False, message=message, data=details, error_code=error_code
    )


def create_success_response(message: str, data: Optional[Any] = None) -> Dict[str, Any]:
    """
    성공 응답 생성

    Args:
        message (str): 성공 메시지
        data (Optional[Any]): 응답 데이터

    Returns:
        Dict[str, Any]: 성공 응답 딕셔너리
    """
    return create_response(success=True, message=message, data=data)
