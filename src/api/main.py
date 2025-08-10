"""
FastAPI 메인 애플리케이션

의료 AI 모델 추론을 위한 REST API 서버입니다.
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import yaml

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.api.routes import predict, health
from src.api.utils.response import create_response
from src.api.utils.validation import validate_config


def create_app() -> FastAPI:
    """
    FastAPI 애플리케이션 생성
    
    Returns:
        FastAPI: 생성된 애플리케이션
    """
    # 애플리케이션 생성
    app = FastAPI(
        title="의료 AI 추론 서비스",
        description="의료 영상 분류를 위한 Attention MIL 모델 API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS 미들웨어 추가
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 라우터 등록
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(predict.router, prefix="/predict", tags=["prediction"])
    
    # 전역 예외 처리
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content=create_response(
                success=False,
                message="서버 내부 오류가 발생했습니다.",
                data=None
            )
        )
    
    # 시작 이벤트
    @app.on_event("startup")
    async def startup_event():
        """서버 시작 시 실행되는 이벤트"""
        print("🚀 의료 AI 추론 서비스 시작...")
        
        # 설정 파일 로드
        try:
            config_path = "configs/model_configs/attention_mil.yaml"
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                validate_config(config)
                print("✅ 설정 파일 로드 완료")
            else:
                print("⚠️ 설정 파일을 찾을 수 없습니다.")
        except Exception as e:
            print(f"❌ 설정 파일 로드 실패: {e}")
        
        # ONNX Runtime 정보 확인
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            print(f"✅ ONNX Runtime 사용 가능")
            print(f"   - 사용 가능한 제공자: {providers}")
            
            if 'CUDAExecutionProvider' in providers:
                print("✅ GPU 가속 지원")
            else:
                print("⚠️ GPU 가속을 사용할 수 없습니다. CPU를 사용합니다.")
                
        except ImportError:
            print("⚠️ ONNX Runtime을 찾을 수 없습니다.")
    
    # 종료 이벤트
    @app.on_event("shutdown")
    async def shutdown_event():
        """서버 종료 시 실행되는 이벤트"""
        print("🛑 의료 AI 추론 서비스 종료...")
    
    return app


# 애플리케이션 인스턴스 생성
app = create_app()


if __name__ == "__main__":
    """개발 서버 실행"""
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 