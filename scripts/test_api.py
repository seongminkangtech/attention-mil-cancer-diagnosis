#!/usr/bin/env python3
"""
FastAPI 서버 테스트 스크립트

의료 AI 추론 서비스의 API 엔드포인트를 테스트합니다.
"""

import requests
import json
import time
from typing import Dict, Any


class APITester:
    """API 테스터 클래스"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health(self) -> Dict[str, Any]:
        """헬스체크 테스트"""
        print("🏥 헬스체크 테스트...")
        
        # 기본 헬스체크
        response = self.session.get(f"{self.base_url}/health/")
        print(f"기본 헬스체크: {response.status_code}")
        if response.status_code == 200:
            print("✅ 서버가 정상 동작 중")
        
        # 상세 헬스체크
        response = self.session.get(f"{self.base_url}/health/detailed")
        print(f"상세 헬스체크: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"시스템 정보: CPU {data['data']['system']['cpu_percent']}%, "
                  f"메모리 {data['data']['system']['memory_percent']}%")
        
        # 준비 상태 확인
        response = self.session.get(f"{self.base_url}/health/ready")
        print(f"준비 상태: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"서비스 준비 상태: {data['data']['ready']}")
        
        return response.json() if response.status_code == 200 else None
    
    def test_model_info(self) -> Dict[str, Any]:
        """모델 정보 테스트"""
        print("\n🤖 모델 정보 테스트...")
        
        response = self.session.get(f"{self.base_url}/predict/model-info")
        print(f"모델 정보: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success', False):
                print(f"모델 이름: {data['data']['model_name']}")
                print(f"클래스 수: {data['data']['num_classes']}")
                print(f"디바이스: {data['data']['device']}")
            else:
                print(f"⚠️ 모델 상태: {data.get('message', '알 수 없는 오류')}")
            return data
        else:
            print("❌ 모델 정보 조회 실패")
            return None
    
    def test_prediction(self, image_paths: list) -> Dict[str, Any]:
        """추론 테스트"""
        print(f"\n🔮 추론 테스트 (이미지 {len(image_paths)}개)...")
        
        # 추론 요청 데이터
        request_data = {
            "image_paths": image_paths,
            "confidence_threshold": 0.5,
            "return_attention": True,
            "batch_size": 1
        }
        
        start_time = time.time()
        response = self.session.post(
            f"{self.base_url}/predict/",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        end_time = time.time()
        
        print(f"추론 요청: {response.status_code}")
        print(f"요청 처리 시간: {end_time - start_time:.2f}초")
        
        if response.status_code == 200:
            data = response.json()
            print(f"예측 결과: {data['data']['prediction']}")
            print(f"신뢰도: {data['data']['confidence']:.3f}")
            print(f"처리 시간: {data['data']['processing_time']:.3f}초")
            
            # 클래스별 확률 출력
            print("클래스별 확률:")
            for class_name, prob in data['data']['class_probabilities'].items():
                print(f"  {class_name}: {prob:.3f}")
            
            return data
        else:
            print(f"❌ 추론 실패: {response.text}")
            return None
    
    def test_model_reload(self) -> Dict[str, Any]:
        """모델 재로드 테스트"""
        print("\n🔄 모델 재로드 테스트...")
        
        response = self.session.post(f"{self.base_url}/predict/reload-model")
        print(f"모델 재로드: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 모델 재로드 성공")
            return data
        else:
            print("❌ 모델 재로드 실패")
            return None
    
    def run_all_tests(self, test_images: list = None):
        """모든 테스트 실행"""
        print("🧪 API 테스트 시작")
        print("=" * 50)
        
        # 헬스체크 테스트
        health_result = self.test_health()
        
        # 모델 정보 테스트
        model_info = self.test_model_info()
        
        # 추론 테스트 (테스트 이미지가 있는 경우)
        if test_images:
            prediction_result = self.test_prediction(test_images)
        else:
            print("\n⚠️ 테스트 이미지가 없어 추론 테스트를 건너뜁니다.")
            prediction_result = None
        
        # 모델 재로드 테스트
        reload_result = self.test_model_reload()
        
        # 결과 요약
        print("\n" + "=" * 50)
        print("📊 테스트 결과 요약")
        print("=" * 50)
        
        tests = [
            ("헬스체크", health_result is not None),
            ("모델 정보", model_info is not None),
            ("추론", prediction_result is not None if test_images else "건너뜀"),
            ("모델 재로드", reload_result is not None)
        ]
        
        for test_name, result in tests:
            status = "✅ 성공" if result else "❌ 실패"
            if result == "건너뜀":
                status = "⏭️ 건너뜀"
            print(f"{test_name}: {status}")
        
        print("=" * 50)


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="API 테스트")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="API 서버 URL (기본값: http://localhost:8000)"
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        help="테스트할 이미지 파일 경로들"
    )
    
    args = parser.parse_args()
    
    # API 테스터 생성
    tester = APITester(args.url)
    
    # 테스트 실행
    tester.run_all_tests(args.images)


if __name__ == "__main__":
    main() 