# Scripts 폴더 - 통합 도구 모음

이 폴더는 Attention MIL 프로젝트의 모든 실행 스크립트를 포함하며, 기능별로 통합되어 있습니다.

## 📁 파일 구조

```
scripts/
├── README.md                    # 이 파일
├── Makefile                     # 간편한 명령어 실행을 위한 Makefile
├── train.py                     # 모델 학습 스크립트
├── onnx_tools.py               # ONNX 관련 모든 도구 (변환, 최적화, 검증)
├── api_tools.py                # API 관련 모든 도구 (서버, 테스트, 모니터링)
├── deployment_tools.py         # 배포 관련 모든 도구 (Docker, K8s, 모델 검증)
└── setup/                      # 설정 관련 스크립트들
```

## 🚀 빠른 시작

### Makefile 사용 (권장)
```bash
# 도움말 보기
make help

# 모델 학습
make train

# ONNX 변환
make onnx-convert

# API 서버 시작
make api-start

# Docker 이미지 빌드
make docker-build
```

### 직접 스크립트 실행
```bash
# ONNX 변환
python scripts/onnx_tools.py --action convert --model-path models/best_model.pth

# API 테스트
python scripts/api_tools.py --action test

# 배포 파이프라인
python scripts/deployment_tools.py --action full-pipeline --model-path models/best_model.pth
```

## 🔧 ONNX 도구 (`onnx_tools.py`)

PyTorch 모델을 ONNX로 변환하고 최적화하는 모든 기능을 제공합니다.

### 주요 기능
- **변환**: PyTorch → ONNX
- **최적화**: 모델 구조 최적화
- **양자화**: INT8 양자화로 모델 크기 감소
- **검증**: ONNX 모델 유효성 검증
- **테스트**: 추론 성능 테스트

### 사용 예제
```bash
# PyTorch 모델을 ONNX로 변환
python scripts/onnx_tools.py --action convert \
    --model-path models/best_model.pth \
    --output-path models/best_model.onnx

# ONNX 모델 최적화
python scripts/onnx_tools.py --action optimize \
    --model-path models/best_model.onnx \
    --output-path models/best_model_optimized.onnx

# ONNX 모델 검증
python scripts/onnx_tools.py --action validate \
    --model-path models/best_model.onnx

# 추론 테스트
python scripts/onnx_tools.py --action test \
    --model-path models/best_model.onnx
```

## 🌐 API 도구 (`api_tools.py`)

FastAPI 서버 관리, 테스트, 성능 측정을 위한 통합 도구입니다.

### 주요 기능
- **서버 관리**: FastAPI 서버 시작/중지
- **API 테스트**: 엔드포인트 테스트 및 검증
- **부하 테스트**: 성능 및 확장성 테스트
- **모니터링**: 지속적인 API 상태 모니터링

### 사용 예제
```bash
# FastAPI 서버 시작
python scripts/api_tools.py --action start \
    --host 0.0.0.0 --port 8000

# 종합 API 테스트
python scripts/api_tools.py --action test

# 부하 테스트 (100 요청, 10 동시)
python scripts/api_tools.py --action load-test \
    --requests 100 --concurrent 10

# 지속 모니터링 (1시간, 1분 간격)
python scripts/api_tools.py --action monitor \
    --monitor-interval 60 --monitor-duration 3600
```

## 🚀 배포 도구 (`deployment_tools.py`)

Docker, Kubernetes, 모델 검증을 위한 종합 배포 도구입니다.

### 주요 기능
- **Docker 관리**: 이미지 빌드, 푸시, 컨테이너 실행
- **Kubernetes 배포**: 네임스페이스 배포, 상태 확인, 스케일링
- **모델 검증**: 성능 측정, 메모리 사용량, 추론 시간
- **전체 파이프라인**: 모델 검증 → Docker 빌드 → K8s 배포

### 사용 예제
```bash
# Docker 이미지 빌드
python scripts/deployment_tools.py --action docker-build \
    --tag attention-mil:latest

# Kubernetes 배포
python scripts/deployment_tools.py --action k8s-deploy \
    --namespace dev

# 배포 상태 확인
python scripts/deployment_tools.py --action k8s-status \
    --namespace dev

# 모델 성능 검증
python scripts/deployment_tools.py --action validate-model \
    --model-path models/best_model.pth

# 전체 배포 파이프라인
python scripts/deployment_tools.py --action full-pipeline \
    --model-path models/best_model.pth \
    --namespace dev
```

## 🎯 모델 학습 (`train.py`)

Attention MIL 모델 학습을 위한 스크립트입니다.

### 주요 기능
- MLflow 통합 로깅
- 하이퍼파라미터 튜닝
- 모델 체크포인트 저장
- 성능 메트릭 추적

### 사용 예제
```bash
# 기본 학습
python scripts/train.py --config configs/attention_mil.yaml

# 커스텀 설정으로 학습
python scripts/train.py \
    --config configs/attention_mil.yaml \
    --experiment-name custom-experiment \
    --epochs 100
```

## 📋 Makefile 명령어

| 명령어 | 설명 |
|--------|------|
| `make help` | 사용 가능한 모든 명령어 표시 |
| `make train` | 모델 학습 실행 |
| `make onnx-convert` | ONNX 변환 |
| `make onnx-optimize` | ONNX 최적화 |
| `make onnx-validate` | ONNX 검증 |
| `make api-start` | API 서버 시작 |
| `make api-test` | API 테스트 |
| `make api-load-test` | API 부하 테스트 |
| `make api-monitor` | API 모니터링 |
| `make docker-build` | Docker 이미지 빌드 |
| `make docker-push` | Docker 이미지 푸시 |
| `make k8s-deploy` | Kubernetes 배포 |
| `make k8s-status` | 배포 상태 확인 |
| `make validate-model` | 모델 성능 검증 |
| `make full-pipeline` | 전체 배포 파이프라인 |
| `make clean` | 임시 파일 정리 |

## 🔧 개발 도구

### 코드 품질
```bash
# 코드 품질 검사
make lint

# 코드 자동 포맷팅
make format

# 테스트 실행
make test
```

### 환경 설정
```bash
# 개발 환경 설정
make setup-dev

# 프로젝트 상태 확인
make status
```

## 📝 설정 파일

모든 스크립트는 `configs/attention_mil.yaml` 설정 파일을 사용합니다. 
새로운 dataclass 기반 설정 시스템과 완벽하게 호환됩니다.

## 🚨 주의사항

1. **의존성**: 필요한 Python 패키지들이 설치되어 있어야 합니다
2. **권한**: Docker 및 Kubernetes 명령어 실행 권한이 필요합니다
3. **설정**: 올바른 설정 파일 경로를 지정해야 합니다
4. **모델 파일**: 모델 파일이 존재하는지 확인하세요

## 🆘 문제 해결

### 일반적인 문제들
- **ImportError**: `PYTHONPATH` 설정 확인
- **권한 오류**: Docker 및 kubectl 권한 확인
- **설정 파일 오류**: YAML 파일 문법 및 경로 확인
- **메모리 부족**: GPU 메모리 또는 시스템 메모리 확인

### 로그 확인
모든 스크립트는 상세한 로그를 출력합니다. 오류 발생 시 로그를 확인하여 문제를 파악하세요.

## 📚 추가 정보

- [프로젝트 개요](../docs/project-overview.md)
- [API 사용법](../docs/api-usage.md)
- [설정 시스템](../configs/README.md)
- [문제 해결](../docs/troubleshooting.md)

