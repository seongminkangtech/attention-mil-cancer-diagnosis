# 📜 스크립트 참조 가이드

## 📋 개요

이 문서는 **Attention MIL Cancer Diagnosis** 프로젝트의 각 스크립트에 대한 상세한 참조 정보를 제공합니다. 각 스크립트의 역할, 사용법, 매개변수, 예제를 포함합니다.

---

## 🚀 **`scripts/train.py` - 모델 학습 스크립트**

### **역할 및 목적**
Attention MIL 모델을 학습시키는 핵심 스크립트입니다. 데이터 로딩부터 모델 저장까지 전체 학습 파이프라인을 관리합니다.

### **주요 기능**
- ✅ YAML 설정 파일 로드 및 검증
- ✅ GPU/CPU 자동 디바이스 선택
- ✅ 데이터 로더 생성 및 전처리
- ✅ 모델 초기화 및 학습
- ✅ MLflow 실험 추적
- ✅ 검증 및 성능 모니터링
- ✅ 최고 성능 모델 자동 저장

### **사용법**
```bash
# 기본 사용법
python scripts/train.py

# 설정 파일 지정
python scripts/train.py --config configs/model_configs/attention_mil.yaml

# 실험 이름 지정
python scripts/train.py --experiment_name "attention_mil_v2_experiment"

# 도움말
python scripts/train.py --help
```

### **명령행 매개변수**
| 매개변수 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `--config` | str | `configs/model_configs/attention_mil.yaml` | 설정 파일 경로 |
| `--experiment_name` | str | `attention_mil_training` | MLflow 실험 이름 |

### **설정 파일 예제**
```yaml
# configs/model_configs/attention_mil.yaml
model:
  num_classes: 3
  feature_extractor:
    model_name: "efficientnet_b2"
    pretrained: true
  attention:
    hidden_dim: 128
    dropout_rate: 0.2

training:
  epochs: 100
  learning_rate: 0.001
  batch_size: 16
  step_size: 30
  gamma: 0.1

data:
  img_size: 256
  image_count: 10
  train_ratio: 0.8
  shuffle: true

hardware:
  device: "auto"  # auto, cuda, cpu

paths:
  frame_path: "data/frames/"
  train_csv: "data/train.csv"
  label_csv: "data/labels.csv"
```

### **출력 예제**
```
사용 디바이스: cuda:0
데이터 로딩 중...
훈련 데이터: 800 샘플
검증 데이터: 200 샘플
학습 시작...
Epoch 1/100
  훈련 - Loss: 1.0986, Acc: 0.3333
  검증 - Loss: 1.0452, Acc: 0.3500
...
학습 완료! 총 소요 시간: 1250.45초
최고 검증 정확도: 0.8950
```

### **MLflow 통합**
- **자동 로깅**: 하이퍼파라미터, 메트릭, 모델 자동 저장
- **실험 추적**: 각 에포크별 성능 지표 기록
- **모델 버전 관리**: 최고 성능 모델 자동 저장

---

## 🔧 **`scripts/api_tools.py` - API 테스트 및 유틸리티**

### **역할 및 목적**
API 서버의 기능을 테스트하고 성능을 벤치마크하는 도구입니다. 개발 및 배포 후 검증에 필수적입니다.

### **주요 기능**
- ✅ API 엔드포인트 상태 확인
- ✅ 추론 성능 벤치마크
- ✅ 배치 추론 테스트
- ✅ 응답 시간 측정
- ✅ 오류 처리 테스트

### **사용법**
```bash
# 기본 테스트 실행
python scripts/api_tools.py

# 특정 엔드포인트 테스트
python scripts/api_tools.py --endpoint health

# 성능 벤치마크
python scripts/api_tools.py --benchmark

# 배치 테스트
python scripts/api_tools.py --batch --size 100
```

### **명령행 매개변수**
| 매개변수 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `--endpoint` | str | `all` | 테스트할 엔드포인트 |
| `--benchmark` | flag | False | 성능 벤치마크 실행 |
| `--batch` | flag | False | 배치 테스트 실행 |
| `--size` | int | 10 | 배치 크기 |
| `--url` | str | `http://localhost:8000` | API 서버 URL |

### **테스트 결과 예제**
```
🔍 API 서버 테스트 시작...
✅ Health Check: /health/status
✅ Predict Endpoint: /predict/cancer
✅ Model Info: /predict/model_info

📊 성능 벤치마크 결과:
  - 평균 응답 시간: 245ms
  - 최대 응답 시간: 890ms
  - 최소 응답 시간: 120ms
  - 처리량: 4.08 req/sec

✅ 모든 테스트 통과!
```

---

## 🚀 **`scripts/deployment_tools.py` - 배포 도구**

### **역할 및 목적**
애플리케이션을 다양한 환경에 배포하기 위한 도구입니다. Docker, Kubernetes, 클라우드 배포를 지원합니다.

### **주요 기능**
- ✅ Docker 이미지 빌드 및 푸시
- ✅ Kubernetes 매니페스트 생성
- ✅ 환경별 설정 관리
- ✅ 배포 상태 모니터링
- ✅ 롤백 및 업그레이드

### **사용법**
```bash
# Docker 이미지 빌드
python scripts/deployment_tools.py --build

# Kubernetes 배포
python scripts/deployment_tools.py --deploy --env dev

# 배포 상태 확인
python scripts/deployment_tools.py --status --env prod

# 롤백
python scripts/deployment_tools.py --rollback --version v1.2.0
```

### **명령행 매개변수**
| 매개변수 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `--build` | flag | False | Docker 이미지 빌드 |
| `--deploy` | flag | False | Kubernetes 배포 |
| `--env` | str | `dev` | 배포 환경 (dev/staging/prod) |
| `--version` | str | `latest` | 배포할 버전 |
| `--rollback` | flag | False | 이전 버전으로 롤백 |

### **배포 파이프라인 예제**
```bash
# 1. 이미지 빌드
python scripts/deployment_tools.py --build --version v1.3.0

# 2. 개발 환경 배포
python scripts/deployment_tools.py --deploy --env dev --version v1.3.0

# 3. 테스트 완료 후 스테이징 배포
python scripts/deployment_tools.py --deploy --env staging --version v1.3.0

# 4. 프로덕션 배포
python scripts/deployment_tools.py --deploy --env prod --version v1.3.0
```

---

## ⚡ **`scripts/onnx_tools.py` - ONNX 변환 및 최적화**

### **역할 및 목적**
PyTorch 모델을 ONNX 형식으로 변환하고 최적화하는 도구입니다. 프로덕션 배포 시 성능 향상을 위해 필수적입니다.

### **주요 기능**
- ✅ PyTorch → ONNX 변환
- ✅ ONNX 모델 검증
- ✅ 성능 최적화
- ✅ 메모리 사용량 최적화
- ✅ 배치 처리 최적화

### **사용법**
```bash
# 기본 변환
python scripts/onnx_tools.py --convert

# 특정 모델 변환
python scripts/onnx_tools.py --convert --model_path models/best_model.pth

# 성능 최적화
python scripts/onnx_tools.py --optimize

# 검증
python scripts/onnx_tools.py --validate --onnx_path models/model.onnx
```

### **명령행 매개변수**
| 매개변수 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `--convert` | flag | False | PyTorch → ONNX 변환 |
| `--optimize` | flag | False | ONNX 모델 최적화 |
| `--validate` | flag | False | ONNX 모델 검증 |
| `--model_path` | str | `models/best_model.pth` | PyTorch 모델 경로 |
| `--onnx_path` | str | `models/model.onnx` | ONNX 모델 경로 |
| `--batch_size` | int | 1 | 배치 크기 |

### **변환 과정 예제**
```
🔄 PyTorch → ONNX 변환 시작...
✅ 모델 로드 완료: models/best_model.pth
✅ ONNX 변환 완료: models/model.onnx
✅ 모델 검증 완료
✅ 성능 테스트 완료

📊 변환 결과:
  - 원본 모델 크기: 45.2 MB
  - ONNX 모델 크기: 44.8 MB
  - 추론 속도 향상: 1.3x
  - 메모리 사용량 감소: 15%
```

---

## 📋 **`scripts/Makefile` - 자동화 명령어**

### **역할 및 목적**
자주 사용하는 명령어들을 간단하게 실행할 수 있도록 도와주는 Makefile입니다. 개발 생산성을 크게 향상시킵니다.

### **사용법**
```bash
# 기본 도움말
make help

# 모델 학습
make train

# 테스트 실행
make test

# Docker 이미지 빌드
make build

# 배포
make deploy

# 정리
make clean
```

### **사용 가능한 명령어들**
| 명령어 | 설명 | 상세 내용 |
|--------|------|-----------|
| `make help` | 사용 가능한 명령어 목록 표시 | 모든 make 명령어와 설명 |
| `make train` | 모델 학습 실행 | `python scripts/train.py` 실행 |
| `make test` | 테스트 실행 | `pytest tests/` 실행 |
| `make lint` | 코드 품질 검사 | Black, Flake8, mypy 실행 |
| `make build` | Docker 이미지 빌드 | `docker build` 실행 |
| `make deploy` | 애플리케이션 배포 | Kubernetes 배포 실행 |
| `make clean` | 임시 파일 정리 | `__pycache__`, `.pytest_cache` 삭제 |
| `make install` | 의존성 설치 | `pip install -r requirements.txt` |
| `make format` | 코드 포맷팅 | Black으로 코드 자동 포맷팅 |

### **Makefile 예제**
```makefile
.PHONY: help train test build deploy clean install format

help:  ## 사용 가능한 명령어 목록 표시
	@echo "사용 가능한 명령어:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

train:  ## 모델 학습 실행
	python scripts/train.py --config configs/model_configs/attention_mil.yaml

test:  ## 테스트 실행
	pytest tests/ -v --cov=src

build:  ## Docker 이미지 빌드
	docker build -t attention-mil:latest .

deploy:  ## 애플리케이션 배포
	python scripts/deployment_tools.py --deploy --env dev

clean:  ## 임시 파일 정리
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf mlruns/
	rm -rf models/*.pth
	rm -rf models/*.onnx

install:  ## 의존성 설치
	pip install -r requirements.txt

format:  ## 코드 포맷팅
	black src/ scripts/ tests/
	isort src/ scripts/ tests/
```

---

## 🔍 **문제 해결 및 디버깅**

### **자주 발생하는 문제들**

#### **1. 모듈 Import 오류**
```bash
# 문제: ModuleNotFoundError: No module named 'src'
# 해결: PYTHONPATH 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/train.py
```

#### **2. CUDA 메모리 부족**
```bash
# 문제: CUDA out of memory
# 해결: 배치 크기 줄이기
python scripts/train.py --config configs/model_configs/attention_mil.yaml
# configs/model_configs/attention_mil.yaml에서 batch_size를 16 → 8로 수정
```

#### **3. MLflow 연결 오류**
```bash
# 문제: MLflow tracking server connection failed
# 해결: MLflow 서버 시작
mlflow server --host 0.0.0.0 --port 5000
```

#### **4. Docker 빌드 실패**
```bash
# 문제: Docker build context too large
# 해결: .dockerignore 파일 확인 및 수정
# 불필요한 파일들 제외 (notebooks/, mlruns/, data/ 등)
```

### **디버깅 팁**

#### **1. 로그 레벨 조정**
```python
# scripts/train.py에서
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### **2. 메모리 사용량 모니터링**
```bash
# GPU 메모리 사용량 확인
nvidia-smi

# 시스템 메모리 확인
htop
```

#### **3. 성능 프로파일링**
```bash
# Python 프로파일링
python -m cProfile -o profile.stats scripts/train.py

# 결과 분석
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(10)"
```

---

## 📚 **추가 리소스**

### **관련 문서**
- [프로젝트 개요](project-overview.md)
- [API 사용법](api-usage.md)
- [개발자 가이드](developer-guide.md)
- [문제 해결 가이드](troubleshooting.md)

### **외부 링크**
- [PyTorch 공식 문서](https://pytorch.org/docs/)
- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [MLflow 공식 문서](https://mlflow.org/docs/)
- [ONNX 공식 문서](https://onnx.ai/)

---

*이 문서는 각 스크립트의 사용법과 문제 해결 방법을 제공합니다. 지속적으로 업데이트됩니다.*
