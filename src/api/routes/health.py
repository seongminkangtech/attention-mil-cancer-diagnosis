"""
헬스체크 엔드포인트

서버 상태 및 모델 상태를 확인하는 API입니다.
"""

import os
import psutil
import torch
from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import Dict, Any

from ..utils.response import create_response

router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """
    기본 헬스체크
    
    Returns:
        Dict[str, Any]: 서버 상태 정보
    """
    return create_response(
        success=True,
        message="서버가 정상적으로 동작 중입니다.",
        data={
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "의료 AI 추론 서비스"
        }
    )


@router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """
    상세 헬스체크
    
    시스템 리소스, GPU 상태, 모델 상태 등을 포함한 상세 정보를 반환합니다.
    
    Returns:
        Dict[str, Any]: 상세 시스템 정보
    """
    try:
        # 시스템 정보
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU 정보
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "available": True,
                "device_name": torch.cuda.get_device_name(0),
                "memory_total": torch.cuda.get_device_properties(0).total_memory,
                "memory_allocated": torch.cuda.memory_allocated(0),
                "memory_cached": torch.cuda.memory_reserved(0)
            }
        else:
            gpu_info = {
                "available": False,
                "message": "GPU를 사용할 수 없습니다."
            }
        
        # 모델 파일 존재 여부 확인
        model_files = []
        config_files = []
        
        # MLflow 모델 확인 (실제 구현에서는 MLflow에서 모델 로드)
        mlruns_path = "mlruns"
        if os.path.exists(mlruns_path):
            model_files.append("MLflow 실험 데이터 존재")
        
        # 설정 파일 확인
        config_path = "configs/model_configs/attention_mil.yaml"
        if os.path.exists(config_path):
            config_files.append("attention_mil.yaml")
        
        return create_response(
            success=True,
            message="상세 시스템 정보",
            data={
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available": memory.available,
                    "memory_total": memory.total,
                    "disk_percent": disk.percent,
                    "disk_free": disk.free,
                    "disk_total": disk.total
                },
                "gpu": gpu_info,
                "model": {
                    "model_files": model_files,
                    "config_files": config_files,
                    "status": "ready" if model_files else "not_ready"
                },
                "timestamp": datetime.now().isoformat()
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"시스템 정보 수집 중 오류 발생: {str(e)}"
        )


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    준비 상태 확인
    
    서비스가 요청을 처리할 준비가 되었는지 확인합니다.
    
    Returns:
        Dict[str, Any]: 준비 상태 정보
    """
    # 기본적인 준비 상태 확인
    checks = {
        "config_file": os.path.exists("configs/model_configs/attention_mil.yaml"),
        "mlruns_exists": os.path.exists("mlruns"),
        "gpu_available": torch.cuda.is_available()
    }
    
    all_ready = all(checks.values())
    
    return create_response(
        success=all_ready,
        message="준비 상태 확인 완료" if all_ready else "일부 구성 요소가 준비되지 않았습니다.",
        data={
            "ready": all_ready,
            "checks": checks,
            "timestamp": datetime.now().isoformat()
        }
    ) 