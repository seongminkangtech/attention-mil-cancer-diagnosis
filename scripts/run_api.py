#!/usr/bin/env python3
"""
FastAPI 서버 실행 스크립트

의료 AI 추론 서비스를 실행합니다.
"""

import os
import sys
import argparse
import uvicorn
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="의료 AI 추론 서비스 실행")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="서버 호스트 (기본값: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="서버 포트 (기본값: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="개발 모드 (자동 리로드)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="워커 프로세스 수 (기본값: 1)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="로그 레벨 (기본값: info)"
    )
    
    args = parser.parse_args()
    
    print("🚀 의료 AI 추론 서비스 시작...")
    print(f"📍 서버 주소: http://{args.host}:{args.port}")
    print(f"📚 API 문서: http://{args.host}:{args.port}/docs")
    print(f"🔄 개발 모드: {args.reload}")
    print(f"👥 워커 수: {args.workers}")
    print(f"📝 로그 레벨: {args.log_level}")
    print("-" * 50)
    
    # 서버 실행
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level,
        access_log=True
    )


if __name__ == "__main__":
    main() 