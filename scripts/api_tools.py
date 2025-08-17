#!/usr/bin/env python3
"""
API 도구 통합 스크립트

FastAPI 서버 실행, 테스트, 성능 측정 등 API 관련 모든 작업을 수행합니다.
"""

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import requests
import uvicorn

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from configs import AppConfig

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIServer:
    """FastAPI 서버 관리 클래스"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.server_process = None

    def start_server(self, host: str = None, port: int = None, workers: int = None):
        """FastAPI 서버 시작"""
        try:
            # 설정에서 기본값 가져오기
            host = host or self.config.fastapi.host
            port = port or self.config.fastapi.port
            workers = workers or self.config.fastapi.workers

            logger.info(f"FastAPI 서버 시작 중... {host}:{port}")

            # uvicorn으로 서버 시작
            uvicorn.run(
                "src.api.main:app",
                host=host,
                port=port,
                reload=self.config.fastapi.reload,
                workers=workers,
                log_level=self.config.fastapi.log_level.lower(),
            )

        except Exception as e:
            logger.error(f"서버 시작 실패: {e}")

    def stop_server(self):
        """서버 중지"""
        if self.server_process:
            self.server_process.terminate()
            logger.info("서버 중지됨")


class APITester:
    """API 테스트 및 성능 측정 클래스"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.base_url = f"http://{config.fastapi.host}:{config.fastapi.port}"

    def test_health_endpoint(self) -> Dict[str, Any]:
        """헬스 체크 엔드포인트 테스트"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            return {
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds(),
                "content": (
                    response.json()
                    if response.headers.get("content-type") == "application/json"
                    else response.text
                ),
            }
        except Exception as e:
            return {"error": str(e)}

    def test_prediction_endpoint(self, image_path: str = None) -> Dict[str, Any]:
        """예측 엔드포인트 테스트"""
        try:
            # 더미 이미지 데이터 생성 (실제 이미지가 없는 경우)
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    files = {"file": f}
                    response = requests.post(
                        f"{self.base_url}/predict", files=files, timeout=30
                    )
            else:
                # 더미 데이터로 테스트
                dummy_data = {
                    "image_data": "dummy_base64_string",
                    "model_name": "attention_mil",
                }
                response = requests.post(
                    f"{self.base_url}/predict", json=dummy_data, timeout=30
                )

            return {
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds(),
                "content": (
                    response.json()
                    if response.headers.get("content-type") == "application/json"
                    else response.text
                ),
            }

        except Exception as e:
            return {"error": str(e)}

    def load_test(
        self, endpoint: str, num_requests: int = 100, concurrent: int = 10
    ) -> Dict[str, Any]:
        """부하 테스트 수행"""
        results = []
        start_time = time.time()

        def make_request():
            try:
                if endpoint == "health":
                    response = requests.get(f"{self.base_url}/health", timeout=5)
                else:
                    response = requests.post(
                        f"{self.base_url}/predict", json={"dummy": "data"}, timeout=30
                    )

                return {
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds(),
                    "success": response.status_code == 200,
                }
            except Exception as e:
                return {"error": str(e), "success": False}

        # 동시 요청 실행
        with ThreadPoolExecutor(max_workers=concurrent) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            for future in futures:
                try:
                    result = future.result(timeout=60)
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e), "success": False})

        end_time = time.time()
        total_time = end_time - start_time

        # 결과 분석
        successful_requests = [r for r in results if r.get("success", False)]
        failed_requests = [r for r in results if not r.get("success", False)]

        response_times = [r.get("response_time", 0) for r in successful_requests]

        return {
            "total_requests": num_requests,
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": len(successful_requests) / num_requests * 100,
            "total_time": total_time,
            "requests_per_second": num_requests / total_time,
            "avg_response_time": (
                statistics.mean(response_times) if response_times else 0
            ),
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "p95_response_time": (
                statistics.quantiles(response_times, n=20)[18]
                if len(response_times) >= 20
                else 0
            ),
        }

    def comprehensive_test(self) -> Dict[str, Any]:
        """종합 API 테스트"""
        logger.info("🔍 종합 API 테스트 시작")

        results = {}

        # 1. 헬스 체크 테스트
        logger.info("1️⃣ 헬스 체크 엔드포인트 테스트")
        health_result = self.test_health_endpoint()
        results["health"] = health_result

        if "error" in health_result:
            logger.error(f"헬스 체크 실패: {health_result['error']}")
            return results

        # 2. 예측 엔드포인트 테스트
        logger.info("2️⃣ 예측 엔드포인트 테스트")
        prediction_result = self.test_prediction_endpoint()
        results["prediction"] = prediction_result

        # 3. 부하 테스트
        logger.info("3️⃣ 부하 테스트 (100 요청, 10 동시)")
        load_result = self.load_test("health", num_requests=100, concurrent=10)
        results["load_test"] = load_result

        # 4. 예측 엔드포인트 부하 테스트
        logger.info("4️⃣ 예측 엔드포인트 부하 테스트 (20 요청, 5 동시)")
        prediction_load_result = self.load_test(
            "predict", num_requests=20, concurrent=5
        )
        results["prediction_load_test"] = prediction_load_result

        return results


class APIMonitor:
    """API 모니터링 클래스"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.tester = APITester(config)

    def continuous_monitoring(self, interval: int = 60, duration: int = 3600):
        """지속적인 API 모니터링"""
        logger.info(
            f"🔄 API 지속 모니터링 시작 (간격: {interval}초, 지속: {duration}초)"
        )

        start_time = time.time()
        monitoring_results = []

        while time.time() - start_time < duration:
            try:
                # 헬스 체크
                health_result = self.tester.test_health_endpoint()

                # 현재 시간과 함께 결과 저장
                result = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "health": health_result,
                }

                monitoring_results.append(result)

                # 결과 출력
                if "error" in health_result:
                    logger.error(
                        f"❌ {result['timestamp']} - API 오류: {health_result['error']}"
                    )
                else:
                    logger.info(
                        f"✅ {result['timestamp']} - API 정상 (응답시간: {health_result['response_time']:.3f}초)"
                    )

                # 대기
                time.sleep(interval)

            except KeyboardInterrupt:
                logger.info("모니터링 중단됨")
                break
            except Exception as e:
                logger.error(f"모니터링 중 오류: {e}")
                time.sleep(interval)

        # 모니터링 결과 요약
        self._print_monitoring_summary(monitoring_results)

        return monitoring_results

    def _print_monitoring_summary(self, results: List[Dict[str, Any]]):
        """모니터링 결과 요약 출력"""
        if not results:
            return

        total_checks = len(results)
        successful_checks = len([r for r in results if "error" not in r["health"]])
        failed_checks = total_checks - successful_checks

        response_times = [
            r["health"].get("response_time", 0)
            for r in results
            if "error" not in r["health"]
        ]

        print("\n" + "=" * 60)
        print("📊 API 모니터링 결과 요약")
        print("=" * 60)
        print(f"총 체크 수: {total_checks}")
        print(f"성공: {successful_checks} ({successful_checks/total_checks*100:.1f}%)")
        print(f"실패: {failed_checks} ({failed_checks/total_checks*100:.1f}%)")

        if response_times:
            print(f"평균 응답시간: {statistics.mean(response_times):.3f}초")
            print(f"최소 응답시간: {min(response_times):.3f}초")
            print(f"최대 응답시간: {max(response_times):.3f}초")

        print("=" * 60)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="API 도구 통합 스크립트")
    parser.add_argument(
        "--action",
        required=True,
        choices=["start", "test", "load-test", "monitor"],
        help="수행할 작업",
    )
    parser.add_argument(
        "--config", default="configs/attention_mil.yaml", help="설정 파일 경로"
    )
    parser.add_argument("--host", help="API 호스트 (기본값: 설정 파일에서)")
    parser.add_argument("--port", type=int, help="API 포트 (기본값: 설정 파일에서)")
    parser.add_argument("--workers", type=int, help="워커 수 (기본값: 설정 파일에서)")
    parser.add_argument("--requests", type=int, default=100, help="부하 테스트 요청 수")
    parser.add_argument(
        "--concurrent", type=int, default=10, help="부하 테스트 동시 요청 수"
    )
    parser.add_argument(
        "--monitor-interval", type=int, default=60, help="모니터링 간격 (초)"
    )
    parser.add_argument(
        "--monitor-duration", type=int, default=3600, help="모니터링 지속 시간 (초)"
    )

    args = parser.parse_args()

    # 설정 로드
    try:
        config = AppConfig.from_yaml(args.config)
    except Exception as e:
        logger.error(f"설정 로드 실패: {e}")
        return

    if args.action == "start":
        # 서버 시작
        server = APIServer(config)
        server.start_server(args.host, args.port, args.workers)

    elif args.action == "test":
        # API 테스트
        tester = APITester(config)
        results = tester.comprehensive_test()

        print("\n📋 API 테스트 결과:")
        print(json.dumps(results, indent=2, ensure_ascii=False))

    elif args.action == "load-test":
        # 부하 테스트
        tester = APITester(config)
        results = tester.load_test("health", args.requests, args.concurrent)

        print("\n📊 부하 테스트 결과:")
        print(json.dumps(results, indent=2, ensure_ascii=False))

    elif args.action == "monitor":
        # 지속 모니터링
        monitor = APIMonitor(config)
        monitor.continuous_monitoring(args.monitor_interval, args.monitor_duration)


if __name__ == "__main__":
    main()
