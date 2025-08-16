# 🚀 GitHub Actions CI/CD 워크플로우 가이드

## 📋 개요

이 디렉토리에는 **Attention MIL Cancer Diagnosis** 프로젝트의 CI/CD 파이프라인을 위한 GitHub Actions 워크플로우 파일들이 포함되어 있습니다.

## 🔧 워크플로우 파일 목록

### 1. **`ci.yml` - 지속적 통합**
- **역할**: 코드 품질 검사, 테스트 실행, 보안 검사
- **트리거**: 모든 브랜치에 푸시, PR 생성 시
- **주요 기능**:
  - 코드 포맷팅 검사 (Black, isort)
  - 코드 품질 검사 (Flake8, MyPy)
  - 단위/통합 테스트 실행
  - 보안 취약점 검사 (Safety, Bandit)
  - Docker 빌드 테스트

### 2. **`cd.yml` - 지속적 배포**
- **역할**: Docker 이미지 빌드, Kubernetes 배포
- **트리거**: main, staging 브랜치에 푸시, 수동 실행
- **주요 기능**:
  - Docker 이미지 빌드 및 푸시
  - 스테이징 환경 배포 (프로덕션 제거)
  - 자동 롤백 준비
  - 배포 결과 알림

### 3. **`model-training.yml` - 모델 학습**
- **역할**: 자동 모델 학습, 검증, 등록
- **트리거**: 매주 일요일 새벽 2시, 수동 실행
- **주요 기능**:
  - 학습 필요성 자동 판단
  - 모델 학습 및 검증
  - ONNX 변환
  - MLflow 모델 등록
  - 새 모델 자동 배포

### 4. **`environment-deploy.yml` - 환경별 배포**
- **역할**: 환경별 차별화된 배포 전략
- **트리거**: develop, staging, main 브랜치에 푸시, 수동 실행
- **주요 기능**:
  - 브랜치별 자동 환경 결정
  - 배포 전 테스트 실행
  - 환경별 배포 방식 차별화
  - 배포 후 기능 테스트

## 🚀 워크플로우 실행 방법

### **자동 실행**
```bash
# 브랜치에 푸시하면 자동으로 실행됩니다
git push origin develop    # CI + 환경별 배포 실행
git push origin staging    # CI + CD + 스테이징 배포 실행
git push origin main       # CI만 실행 (프로덕션 배포 제거됨)
```

### **수동 실행**
1. GitHub 저장소의 **Actions** 탭으로 이동
2. 원하는 워크플로우 선택
3. **Run workflow** 버튼 클릭
4. 필요한 매개변수 입력 후 실행

## 🔐 필요한 GitHub Secrets

### **필수 시크릿**
```bash
# Kubernetes 설정 (base64 인코딩)
KUBECONFIG_STAGING        # 스테이징 환경 kubeconfig

# MLflow 설정
MLFLOW_TRACKING_URI       # MLflow 서버 URI
MLFLOW_TRACKING_USERNAME  # MLflow 사용자명
MLFLOW_TRACKING_PASSWORD  # MLflow 비밀번호
```

### **선택적 시크릿**
```bash
# 알림 및 모니터링
SLACK_WEBHOOK_URL         # Slack 알림 웹훅
EMAIL_SMTP_PASSWORD       # 이메일 알림 SMTP 비밀번호

# 보안 스캔
SNYK_TOKEN                # Snyk 보안 스캔 토큰
```

## 🏗️ 환경별 배포 전략

### **개발 환경 (develop)**
- **배포 방식**: Docker Compose
- **테스트**: 기본 테스트만 실행
- **자동화**: 브랜치 푸시 시 자동 배포

### **스테이징 환경 (staging)**
- **배포 방식**: Kubernetes
- **테스트**: 전체 테스트 스위트 실행
- **자동화**: 브랜치 푸시 시 자동 배포
- **검증**: 수동 승인 후 프로덕션 배포

### **프로덕션 환경 (main)**
- **배포 방식**: ❌ 제거됨 (스테이징까지만 구성)
- **테스트**: ❌ 제거됨
- **자동화**: ❌ 제거됨
- **모니터링**: ❌ 제거됨

## 📊 워크플로우 상태 모니터링

### **GitHub Actions 대시보드**
- **Actions** 탭에서 모든 워크플로우 실행 상태 확인
- **실시간 로그** 스트리밍
- **실행 시간** 및 **리소스 사용량** 통계

### **알림 설정**
```yaml
# .github/workflows/notifications.yml 예시
- name: Slack 알림
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    channel: '#deployments'
    webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
```

## 🔍 문제 해결

### **자주 발생하는 문제들**

#### **1. CI 실패**
```bash
# 로컬에서 테스트 실행
make test
make lint
make format
```

#### **2. CD 실패**
```bash
# Docker 이미지 로컬 빌드 테스트
docker build -t attention-mil:test .
docker run -p 8000:8000 attention-mil:test
```

#### **3. 모델 학습 실패**
```bash
# MLflow 서버 상태 확인
mlflow server --host 0.0.0.0 --port 5000

# 로컬에서 학습 테스트
python scripts/train.py --config configs/model_configs/attention_mil.yaml
```

### **디버깅 팁**
1. **Actions** 탭에서 실패한 단계의 로그 확인
2. **로컬 환경**에서 동일한 명령어 실행
3. **의존성 버전** 호환성 확인
4. **시크릿 설정** 올바른지 확인

## 📈 성능 최적화

### **캐싱 전략**
```yaml
# Python 의존성 캐싱
- uses: actions/setup-python@v4
  with:
    cache: 'pip'

# Docker 레이어 캐싱
- uses: docker/build-push-action@v5
  with:
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

### **병렬 실행**
```yaml
# 독립적인 작업들을 병렬로 실행
jobs:
  code-quality:
    # 코드 품질 검사
  test:
    # 테스트 실행
  security:
    # 보안 검사
```

## 🔄 워크플로우 업데이트

### **워크플로우 수정 시 주의사항**
1. **문법 검증**: YAML 문법 오류 확인
2. **의존성 관리**: 작업 간 의존성 올바르게 설정
3. **권한 설정**: 필요한 권한이 있는지 확인
4. **테스트**: 수정된 워크플로우 로컬 테스트

### **버전 관리**
```bash
# 워크플로우 변경사항 커밋
git add .github/workflows/
git commit -m "ci: 워크플로우 업데이트 - 새로운 테스트 추가"
git push origin main
```

## 📚 추가 리소스

### **GitHub Actions 공식 문서**
- [GitHub Actions 기본 개념](https://docs.github.com/en/actions/learn-github-actions)
- [워크플로우 문법](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [환경 설정](https://docs.github.com/en/actions/deployment/targeting-different-environments)

### **관련 도구 문서**
- [Docker GitHub Actions](https://github.com/docker/build-push-action)
- [Kubernetes GitHub Actions](https://github.com/azure/setup-kubectl)
- [MLflow GitHub Actions](https://mlflow.org/docs/latest/tracking.html)

---

*이 가이드는 GitHub Actions 워크플로우의 사용법과 문제 해결 방법을 제공합니다. 지속적으로 업데이트됩니다.*
