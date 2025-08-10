#!/usr/bin/env python3
"""
Docker 빌드 및 실행 스크립트

의료 AI 추론 서비스의 Docker 컨테이너를 빌드하고 실행합니다.
"""

import subprocess
import sys
import argparse
import os
from pathlib import Path


def run_command(command: str, check: bool = True) -> bool:
    """명령어 실행"""
    try:
        print(f"🔄 실행 중: {command}")
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 오류 발생: {e}")
        if e.stderr:
            print(f"오류 메시지: {e.stderr}")
        return False


def build_image(tag: str = "medical-ai-api") -> bool:
    """Docker 이미지 빌드"""
    print("🏗️ Docker 이미지 빌드 중...")
    command = f"docker build -t {tag} ."
    return run_command(command)


def run_container(tag: str = "medical-ai-api", port: int = 8000, mode: str = "dev") -> bool:
    """Docker 컨테이너 실행"""
    print(f"🚀 Docker 컨테이너 실행 중... (모드: {mode})")
    
    if mode == "dev":
        # 개발 모드: 볼륨 마운트, 자동 리로드
        command = f"""docker run -d --name medical-ai-api-dev \
            -p {port}:8000 \
            -v {os.getcwd()}:/app \
            -e PYTHONPATH=/app \
            -e ENVIRONMENT=development \
            {tag} \
            python scripts/run_api.py --host 0.0.0.0 --port 8000 --reload"""
    else:
        # 프로덕션 모드
        command = f"""docker run -d --name medical-ai-api-prod \
            -p {port}:8000 \
            -e PYTHONPATH=/app \
            -e ENVIRONMENT=production \
            {tag} \
            python scripts/run_api.py --host 0.0.0.0 --port 8000 --workers 4"""
    
    return run_command(command)


def stop_container(container_name: str) -> bool:
    """Docker 컨테이너 중지"""
    print(f"🛑 컨테이너 중지 중: {container_name}")
    command = f"docker stop {container_name}"
    return run_command(command)


def remove_container(container_name: str) -> bool:
    """Docker 컨테이너 제거"""
    print(f"🗑️ 컨테이너 제거 중: {container_name}")
    command = f"docker rm {container_name}"
    return run_command(command, check=False)  # 컨테이너가 없어도 오류가 아님


def docker_compose_up(profile: str = "dev") -> bool:
    """Docker Compose로 서비스 실행"""
    print(f"🚀 Docker Compose 실행 중... (프로필: {profile})")
    command = f"docker-compose --profile {profile} up -d"
    return run_command(command)


def docker_compose_down() -> bool:
    """Docker Compose 서비스 중지"""
    print("🛑 Docker Compose 서비스 중지 중...")
    command = "docker-compose down"
    return run_command(command)


def show_logs(container_name: str = "medical-ai-api-dev") -> bool:
    """컨테이너 로그 출력"""
    print(f"📋 컨테이너 로그: {container_name}")
    command = f"docker logs -f {container_name}"
    return run_command(command)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Docker 빌드 및 실행")
    parser.add_argument(
        "--action",
        type=str,
        choices=["build", "run", "stop", "logs", "compose-up", "compose-down"],
        default="build",
        help="실행할 작업"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="medical-ai-api",
        help="Docker 이미지 태그"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="포트 번호"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dev", "prod"],
        default="dev",
        help="실행 모드"
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["dev", "prod", "test"],
        default="dev",
        help="Docker Compose 프로필"
    )
    parser.add_argument(
        "--container",
        type=str,
        default="medical-ai-api-dev",
        help="컨테이너 이름"
    )

    args = parser.parse_args()

    if args.action == "build":
        success = build_image(args.tag)
        if success:
            print("✅ Docker 이미지 빌드 완료")
        else:
            print("❌ Docker 이미지 빌드 실패")
            sys.exit(1)

    elif args.action == "run":
        # 기존 컨테이너 정리
        stop_container(args.container)
        remove_container(args.container)
        
        # 새 컨테이너 실행
        success = run_container(args.tag, args.port, args.mode)
        if success:
            print("✅ Docker 컨테이너 실행 완료")
            print(f"🌐 서버 주소: http://localhost:{args.port}")
            print(f"📚 API 문서: http://localhost:{args.port}/docs")
        else:
            print("❌ Docker 컨테이너 실행 실패")
            sys.exit(1)

    elif args.action == "stop":
        success = stop_container(args.container)
        if success:
            print("✅ 컨테이너 중지 완료")

    elif args.action == "logs":
        show_logs(args.container)

    elif args.action == "compose-up":
        success = docker_compose_up(args.profile)
        if success:
            print("✅ Docker Compose 서비스 실행 완료")
        else:
            print("❌ Docker Compose 서비스 실행 실패")
            sys.exit(1)

    elif args.action == "compose-down":
        success = docker_compose_down()
        if success:
            print("✅ Docker Compose 서비스 중지 완료")


if __name__ == "__main__":
    main() 