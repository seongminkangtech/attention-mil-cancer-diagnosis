"""
설정 사용 예제

새로운 dataclass 기반 설정 시스템의 사용법을 보여줍니다.
"""

import os

from configs import AppConfig
from configs.validators import comprehensive_validation


def example_basic_usage():
    """기본 사용법 예제"""
    print("🔧 기본 설정 사용법")
    print("=" * 50)

    # 1. 환경 변수에서 설정 로드
    config = AppConfig.from_env()

    # 2. 설정 요약 출력
    config.print_summary()

    # 3. 개별 설정 접근
    print(f"\n📊 개별 설정 접근:")
    print(f"   - 환경: {config.environment.env.value}")
    print(f"   - 모델 클래스 수: {config.attention_mil.num_classes}")
    print(f"   - API 포트: {config.fastapi.port}")
    print(f"   - MLflow 실험: {config.mlflow.experiment_name}")


def example_yaml_compatibility():
    """YAML 호환성 예제"""
    print("\n📄 YAML 호환성 사용법")
    print("=" * 50)

    # YAML 파일에서 설정 로드 (하위 호환성)
    yaml_path = "configs/model_configs/attention_mil.yaml"
    if os.path.exists(yaml_path):
        config = AppConfig.from_yaml(yaml_path)
        print("✅ YAML 파일에서 설정 로드 성공")

        # 설정 검증
        validation_results = comprehensive_validation(config)
        print(f"🔍 설정 검증 결과: {validation_results}")
    else:
        print("⚠️ YAML 파일을 찾을 수 없습니다.")


def example_environment_specific():
    """환경별 설정 예제"""
    print("\n🌍 환경별 설정 예제")
    print("=" * 50)

    # 개발 환경
    os.environ["ENVIRONMENT"] = "development"
    dev_config = AppConfig.from_env()
    print(f"🔧 개발 환경:")
    print(f"   - 디버그 모드: {dev_config.environment.debug}")
    print(f"   - 워커 수: {dev_config.fastapi.workers}")
    print(f"   - 자동 리로드: {dev_config.fastapi.reload}")

    # 프로덕션 환경
    os.environ["ENVIRONMENT"] = "production"
    prod_config = AppConfig.from_env()
    print(f"\n🚀 프로덕션 환경:")
    print(f"   - 디버그 모드: {prod_config.environment.debug}")
    print(f"   - 워커 수: {prod_config.fastapi.workers}")
    print(f"   - 자동 리로드: {prod_config.fastapi.reload}")

    # 환경 변수 복원
    os.environ["ENVIRONMENT"] = "development"


def example_custom_config():
    """사용자 정의 설정 예제"""
    print("\n⚙️ 사용자 정의 설정 예제")
    print("=" * 50)

    # 기본 설정 로드
    config = AppConfig.from_env()

    # 설정 수정
    config.attention_mil.num_classes = 5
    config.attention_mil.training.epochs = 100
    config.attention_mil.training.learning_rate = 1e-3
    config.fastapi.port = 9000

    print("✅ 사용자 정의 설정 적용:")
    print(f"   - 클래스 수: {config.attention_mil.num_classes}")
    print(f"   - 에포크 수: {config.attention_mil.training.epochs}")
    print(f"   - 학습률: {config.attention_mil.training.learning_rate}")
    print(f"   - API 포트: {config.fastapi.port}")


def example_validation():
    """설정 검증 예제"""
    print("\n🔍 설정 검증 예제")
    print("=" * 50)

    # 설정 로드
    config = AppConfig.from_env()

    # 종합 검증
    validation_results = comprehensive_validation(config)

    print("🔍 검증 결과:")
    for check, result in validation_results.items():
        status = "✅" if result else "❌"
        print(f"   {status} {check}: {result}")

    if validation_results["overall"]:
        print("\n🎉 모든 검증을 통과했습니다!")
    else:
        print("\n⚠️ 일부 검증에 실패했습니다.")


def example_mlflow_integration():
    """MLflow 통합 예제"""
    print("\n📊 MLflow 통합 예제")
    print("=" * 50)

    # 설정 로드
    config = AppConfig.from_env()

    # MLflow 설정 확인
    print("🔧 MLflow 설정:")
    print(f"   - 추적 URI: {config.mlflow.tracking_uri}")
    print(f"   - 레지스트리 URI: {config.mlflow.registry_uri}")
    print(f"   - 실험 이름: {config.mlflow.experiment_name}")
    print(f"   - 모델 레지스트리: {config.mlflow.model_registry_name}")

    # 환경 변수 설정
    os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow-server:5000"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "custom-experiment"

    # 새로운 설정 로드
    new_config = AppConfig.from_env()
    print(f"\n🔄 업데이트된 MLflow 설정:")
    print(f"   - 추적 URI: {new_config.mlflow.tracking_uri}")
    print(f"   - 실험 이름: {new_config.mlflow.experiment_name}")


def main():
    """메인 함수"""
    print("🚀 의료 AI 프로젝트 설정 시스템 예제")
    print("=" * 60)

    try:
        # 각 예제 실행
        example_basic_usage()
        example_yaml_compatibility()
        example_environment_specific()
        example_custom_config()
        example_validation()
        example_mlflow_integration()

        print("\n🎉 모든 예제가 성공적으로 실행되었습니다!")

    except Exception as e:
        print(f"\n❌ 예제 실행 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
