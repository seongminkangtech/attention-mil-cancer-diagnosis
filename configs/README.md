# 🔧 의료 AI 프로젝트 설정 시스템

## 📋 개요

이 프로젝트는 **dataclass 기반의 타입 안전한 설정 관리 시스템**을 제공합니다. 기존의 YAML 파일 기반 설정보다 훨씬 안전하고 유지보수하기 쉽습니다.

## ✨ 주요 특징

### 🎯 **타입 안전성**
- **런타임 에러 방지**: 설정 오류를 컴파일 타임에 감지
- **IDE 지원**: 자동완성, 타입 체크, 리팩토링 지원
- **mypy 검증**: 정적 타입 검사로 코드 품질 향상

### 🔧 **유지보수성**
- **중복 설정 방지**: 계층적 구조로 명확한 구분
- **자동 검증**: 설정 변경 시 즉시 유효성 검사
- **문서화**: 코드와 문서가 자동으로 일치

### 🌍 **환경별 관리**
- **자동 환경 감지**: 개발/스테이징/프로덕션 환경별 설정
- **환경 변수 통합**: 시스템 환경 변수와 자동 동기화
- **보안 강화**: 프로덕션 환경에서 자동 보안 설정

## 📁 구조

```
configs/
├── __init__.py                 # 패키지 초기화
├── base_config.py              # 기본 설정 클래스들
├── environment_config.py        # 환경별 설정
├── app_config.py               # 통합 설정
├── validators.py               # 설정 검증
├── example_usage.py            # 사용 예제
├── README.md                   # 이 문서
├── model_configs/              # 모델별 설정
│   ├── __init__.py
│   └── attention_mil_config.py
└── api_configs/                # API 설정
    ├── __init__.py
    └── fastapi_config.py
```

## 🚀 사용법

### 1. **기본 사용법**

```python
from configs import AppConfig

# 환경 변수에서 설정 로드
config = AppConfig.from_env()

# 설정 사용
print(f"환경: {config.environment.env.value}")
print(f"API 포트: {config.fastapi.port}")
print(f"모델 클래스 수: {config.attention_mil.num_classes}")
```

### 2. **YAML 호환성 (하위 호환성)**

```python
# 기존 YAML 파일에서 설정 로드
config = AppConfig.from_yaml("configs/model_configs/attention_mil.yaml")

# 설정 검증
from configs.validators import comprehensive_validation
results = comprehensive_validation(config)
```

### 3. **환경별 설정**

```python
# 환경 변수 설정
os.environ["ENVIRONMENT"] = "production"

# 환경별 설정 자동 적용
config = AppConfig.from_env()

# 프로덕션 환경에서는 자동으로:
# - 디버그 모드 비활성화
# - 워커 수 증가
# - 보안 설정 강화
```

### 4. **설정 검증**

```python
from configs.validators import comprehensive_validation

# 종합적인 설정 검증
validation_results = comprehensive_validation(config)

# 검증 결과 확인
if validation_results['overall']:
    print("✅ 모든 검증을 통과했습니다!")
else:
    print("❌ 일부 검증에 실패했습니다.")
```

## 🔧 설정 클래스들

### **EnvironmentConfig**
- 환경 타입 (development, staging, production, test)
- 디버그 모드, 로그 레벨, 워커 수 등

### **AttentionMILConfig**
- 모델 아키텍처 설정
- 학습 파라미터, 데이터 설정, 배포 설정 등

### **FastAPIConfig**
- 서버 설정 (호스트, 포트, 워커 등)
- CORS, 미들웨어, 보안 설정 등

### **MLflowConfig**
- 실험 추적 설정
- 모델 레지스트리, 아티팩트 저장 등

## 🌍 환경별 설정

### **개발 환경 (Development)**
```python
# 기본값
debug = True
workers = 1
log_level = "DEBUG"
cors_origins = ["*"]
```

### **스테이징 환경 (Staging)**
```python
# 중간 보안
debug = False
workers = 2
log_level = "INFO"
cors_origins = ["https://staging.yourdomain.com"]
```

### **프로덕션 환경 (Production)**
```python
# 최고 보안
debug = False
workers = 4
log_level = "WARNING"
cors_origins = ["https://yourdomain.com"]
```

## 🔍 설정 검증

### **자동 검증 항목**
1. **기본 검증**: 타입, 범위, 필수 값 확인
2. **경로 검증**: 파일/디렉토리 존재 여부
3. **환경별 검증**: 프로덕션 환경 보안 설정
4. **모델 검증**: 모델 파라미터 유효성
5. **API 검증**: 서버 설정 유효성

### **검증 예제**

```python
# 개별 검증
from configs.validators import (
    validate_config,
    validate_paths,
    validate_environment_config,
    validate_model_config,
    validate_api_config
)

# 전체 검증
results = comprehensive_validation(config)
print(f"검증 결과: {results}")
```

## 📊 MLflow 통합

### **자동 설정**
```python
# 환경 변수에서 자동 로드
MLFLOW_TRACKING_URI=http://mlflow-server:5000
MLFLOW_EXPERIMENT_NAME=custom-experiment
MLFLOW_TRACKING_USERNAME=user
MLFLOW_TRACKING_PASSWORD=pass
```

### **설정 사용**
```python
config = AppConfig.from_env()

# MLflow 설정 확인
print(f"추적 URI: {config.mlflow.tracking_uri}")
print(f"실험 이름: {config.mlflow.experiment_name}")
```

## 🔄 마이그레이션 가이드

### **기존 YAML 사용자**
1. **단계적 전환**: `AppConfig.from_yaml()` 사용
2. **설정 검증**: `comprehensive_validation()` 추가
3. **점진적 업데이트**: dataclass 기반으로 단계적 전환

### **환경 변수 사용자**
1. **즉시 사용**: `AppConfig.from_env()` 사용
2. **자동 동기화**: 환경 변수와 설정 자동 동기화
3. **환경별 최적화**: 자동 환경별 설정 적용

## 🧪 테스트

### **설정 검증 테스트**
```bash
# 예제 실행
python configs/example_usage.py

# 설정 검증
python -c "from configs import AppConfig; config = AppConfig.from_env(); print('✅ 설정 로드 성공')"
```

### **단위 테스트**
```bash
# pytest로 테스트 실행
pytest tests/unit/test_configs.py -v
```

## 🚨 주의사항

### **프로덕션 환경**
- `SECRET_KEY` 반드시 설정
- CORS를 `*`로 설정 금지
- 로그 레벨을 `DEBUG`로 설정 금지

### **환경 변수**
- 민감한 정보는 환경 변수로 관리
- `.env` 파일을 버전 관리에 포함 금지
- 기본값은 개발 환경용으로만 사용

## 📚 추가 자료

- [Python dataclasses 공식 문서](https://docs.python.org/3/library/dataclasses.html)
- [MLflow 설정 가이드](https://mlflow.org/docs/latest/tracking.html)
- [FastAPI 설정 가이드](https://fastapi.tiangolo.com/advanced/settings/)

## 🤝 기여하기

설정 시스템 개선에 기여하고 싶으시다면:

1. **이슈 등록**: 버그나 개선 사항 보고
2. **풀 리퀘스트**: 새로운 기능이나 수정 사항 제안
3. **문서 개선**: 사용법이나 예제 추가

---

**🎯 목표**: 타입 안전하고 유지보수하기 쉬운 설정 시스템으로 의료 AI 프로젝트의 안정성과 개발자 경험을 향상시키기
