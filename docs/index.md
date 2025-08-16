# 📚 Attention MIL Cancer Diagnosis - 문서 인덱스

## 🎯 프로젝트 소개

**Attention MIL Cancer Diagnosis**는 Multiple Instance Learning과 Attention Mechanism을 결합하여 병리학적 이미지에서 암을 진단하는 의료 AI 프로젝트입니다.

---

## 📖 문서 목록

### 🏠 **프로젝트 개요**
- **[프로젝트 종합 개요](project-overview.md)** - 프로젝트의 전체적인 소개, 아키텍처, 기술 스택, 로드맵

### 🧑‍💻 **개발자 가이드**
- **[개발자 가이드](developer-guide.md)** - 프로젝트 구조 및 각 폴더/스크립트의 역할 상세 설명
- **[스크립트 참조 가이드](scripts-reference.md)** - 각 스크립트의 사용법, 매개변수, 예제

### 🔧 **사용법 가이드**
- **[API 사용법](api-usage.md)** - REST API 엔드포인트 사용법 및 예제
- **[결과 해석 가이드](result-interpretation.md)** - 모델 출력 결과 해석 방법

### 🚀 **배포 및 운영**
- **[CI/CD 파이프라인](ci-cd-pipeline.md)** - 자동화된 빌드, 테스트, 배포 파이프라인
- **[문제 해결 가이드](troubleshooting.md)** - 자주 발생하는 문제들과 해결 방법

### 📋 **컨벤션 및 가이드**
- **[Git 컨벤션](git-conventions.md)** - 브랜치 전략, 커밋 메시지, PR 가이드
- **[README 작성 가이드](README_guide.md)** - README.md 작성 시 참고할 템플릿과 가이드
- **[디렉토리 구조 가이드](directory.md)** - 프로젝트 폴더 구조 설계 원칙

### ❓ **FAQ 및 지원**
- **[자주 묻는 질문](faq.md)** - 프로젝트 사용 시 자주 묻는 질문들과 답변

---

## 🏗️ 프로젝트 구조 요약

```
attention-mil/
├── 📁 src/                    # 핵심 소스 코드
│   ├── 📁 api/               # FastAPI 서버 및 라우터
│   ├── 📁 models/            # AI 모델 구현
│   ├── 📁 utils/             # 공통 유틸리티
│   └── 📁 mlops/             # MLOps 관련 (향후 확장)
├── 📁 configs/                # 설정 관리
│   ├── 📁 api_configs/       # API 서버 설정
│   ├── 📁 model_configs/     # 모델 설정
│   └── 📁 base_config.py     # 공통 설정 클래스
├── 📁 scripts/                # 실행 스크립트
│   ├── 📄 train.py           # 모델 학습 (핵심)
│   ├── 📄 api_tools.py       # API 테스트 도구
│   ├── 📄 deployment_tools.py # 배포 도구
│   ├── 📄 onnx_tools.py      # ONNX 변환 도구
│   └── 📄 Makefile           # 자동화 명령어
├── 📁 tests/                  # 테스트 코드
│   ├── 📁 unit/              # 단위 테스트
│   ├── 📁 integration/       # 통합 테스트
│   └── 📁 fixtures/          # 테스트 데이터
├── 📁 docs/                   # 프로젝트 문서
├── 📁 k8s/                    # Kubernetes 배포
├── 📁 docker/                 # Docker 관련
└── 📁 notebooks/              # 실험 및 분석 노트북
```

---

## 🚀 빠른 시작 가이드

### **1. 프로젝트 클론 및 설정**
```bash
git clone https://github.com/your-org/attention-mil.git
cd attention-mil
pip install -r requirements.txt
```

### **2. 모델 학습**
```bash
# 기본 학습 실행
make train

# 또는 직접 실행
python scripts/train.py --config configs/model_configs/attention_mil.yaml
```

### **3. API 서버 실행**
```bash
# 개발 모드로 실행
python -m uvicorn src.api.main:app --reload

# 또는 Docker로 실행
docker build -t attention-mil:latest .
docker run -p 8000:8000 attention-mil:latest
```

### **4. 테스트 실행**
```bash
# 모든 테스트 실행
make test

# 특정 테스트만 실행
pytest tests/unit/test_models.py -v
```

---

## 📚 학습 경로

### **👶 초급자 (의료진, 연구자)**
1. **[프로젝트 개요](project-overview.md)** 읽기
2. **[API 사용법](api-usage.md)** 학습
3. **[결과 해석 가이드](result-interpretation.md)** 참조

### **👨‍💻 중급자 (개발자)**
1. **[개발자 가이드](developer-guide.md)** 상세 학습
2. **[스크립트 참조 가이드](scripts-reference.md)** 참조
3. 코드 구조 탐색 및 수정

### **👨‍🔬 고급자 (MLOps 엔지니어)**
1. **[CI/CD 파이프라인](ci-cd-pipeline.md)** 분석
2. **[문제 해결 가이드](troubleshooting.md)** 참조
3. 배포 및 모니터링 시스템 구축

---

## 🔍 문서 검색

### **주요 키워드**
- **모델 학습**: `train.py`, `MLflow`, `하이퍼파라미터`
- **API 개발**: `FastAPI`, `라우터`, `미들웨어`
- **배포**: `Docker`, `Kubernetes`, `CI/CD`
- **문제 해결**: `디버깅`, `오류`, `성능 최적화`

### **문서별 주요 내용**
| 문서 | 주요 내용 | 대상 독자 |
|------|-----------|-----------|
| [프로젝트 개요](project-overview.md) | 전체 시스템 구조, 기술 스택, 로드맵 | 모든 사용자 |
| [개발자 가이드](developer-guide.md) | 코드 구조, 폴더 역할, 개발 워크플로우 | 개발자 |
| [스크립트 참조](scripts-reference.md) | 스크립트 사용법, 매개변수, 예제 | 개발자, DevOps |
| [API 사용법](api-usage.md) | API 엔드포인트, 요청/응답 형식 | API 사용자 |
| [CI/CD 파이프라인](ci-cd-pipeline.md) | 자동화, 배포, 모니터링 | MLOps 엔지니어 |

---

## 📞 지원 및 기여

### **문서 개선**
- 문서에 오류나 개선사항이 있다면 [GitHub Issues](https://github.com/your-org/attention-mil/issues)에 등록
- 직접 수정하고 PR을 보내는 것도 환영

### **질문 및 토론**
- 일반적인 질문: [GitHub Discussions](https://github.com/your-org/attention-mil/discussions)
- 버그 리포트: [GitHub Issues](https://github.com/your-org/attention-mil/issues)

### **기여 가이드**
- [Git 컨벤션](git-conventions.md) 준수
- [README 작성 가이드](README_guide.md) 참조
- 모든 코드 변경에 대한 테스트 작성

---

## 📈 문서 업데이트 이력

| 날짜 | 버전 | 주요 변경사항 |
|------|------|---------------|
| 2024-12-XX | v1.0.0 | 초기 문서 작성 |
| - | - | 개발자 가이드 추가 |
| - | - | 스크립트 참조 가이드 추가 |

---

## 🔗 관련 링크

### **프로젝트 관련**
- [GitHub 저장소](https://github.com/your-org/attention-mil)
- [이슈 트래커](https://github.com/your-org/attention-mil/issues)
- [위키](https://github.com/your-org/attention-mil/wiki)

### **외부 리소스**
- [PyTorch 공식 문서](https://pytorch.org/docs/)
- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [MLflow 공식 문서](https://mlflow.org/docs/)
- [Kubernetes 공식 문서](https://kubernetes.io/docs/)

---

*이 문서는 프로젝트의 모든 문서를 체계적으로 정리한 인덱스입니다. 필요한 정보를 빠르게 찾을 수 있도록 구성되었습니다.*
