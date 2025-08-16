# 🚀 GitHub Actions 워크플로우 가이드

이 문서는 Attention MIL 프로젝트의 GitHub Actions 워크플로우 사용법과 최적화 내용을 설명합니다.

## 📋 워크플로우 개요

### 1. **CI (지속적 통합)**
- **파일**: `.github/workflows/ci.yml`
- **트리거**: 모든 브랜치 푸시, PR
- **목적**: 코드 품질 검사, 테스트, 보안 검사

### 2. **CD (지속적 배포)**
- **파일**: `.github/workflows/cd.yml`
- **트리거**: main/staging 브랜치 푸시
- **목적**: Docker 이미지 빌드, Kubernetes 배포

### 3. **릴리스**
- **파일**: `.github/workflows/release.yml`
- **트리거**: 태그 푸시 (v*)
- **목적**: GitHub 릴리스 생성, 이미지 태깅

### 4. **의존성 검사**
- **파일**: `.github/workflows/dependency-review.yml`
- **트리거**: PR 생성/업데이트
- **목적**: 보안 취약점 검사

## 🔧 워크플로우 최적화 특징

### **병렬 실행**
- 독립적인 작업들을 병렬로 실행하여 전체 시간 단축
- CI 단계: 테스트, 보안 검사, Docker 빌드 병렬 실행

### **캐시 최적화**
- Docker 레이어 캐시 (GitHub Actions 캐시 활용)
- Python 의존성 캐시 (pip 캐시)
- 빌드 컨텍스트 최적화

### **보안 강화**
- Trivy를 통한 컨테이너 보안 스캔
- CodeQL을 통한 코드 보안 분석
- 의존성 보안 취약점 검사

### **에러 처리**
- 자동 롤백 스크립트 생성
- 실패 시 알림 및 가이드 제공
- 단계별 상태 모니터링

## 🚀 사용 방법

### **자동 실행**
```bash
# 브랜치 푸시 시 자동 실행
git push origin main

# PR 생성 시 자동 실행
git push origin feature/new-feature
```

### **수동 실행**
```bash
# GitHub Actions 페이지에서 수동 실행
# Actions > CD - 지속적 배포 > Run workflow

# 환경 선택
# - staging: 스테이징 환경 배포
# - production: 프로덕션 환경 배포
```

### **릴리스 배포**
```bash
# 태그 생성 및 푸시
git tag v1.2.0
git push origin v1.2.0

# 자동으로 릴리스 생성 및 배포
```

## 📊 워크플로우 단계별 설명

### **CI 워크플로우**
1. **테스트 실행**
   - Python 3.9, 3.10 환경에서 테스트
   - 커버리지 리포트 생성
   - 결과 아티팩트 업로드

2. **보안 검사**
   - Safety: 의존성 보안 검사
   - Bandit: 코드 보안 검사
   - 결과 리포트 생성

3. **Docker 빌드**
   - 테스트 이미지 빌드
   - 보안 스캔 실행
   - 빌드 검증

4. **품질 게이트**
   - 모든 검사 결과 확인
   - 품질 기준 충족 여부 검증

### **CD 워크플로우**
1. **이미지 빌드 및 푸시**
   - 멀티 플랫폼 빌드 (amd64, arm64)
   - GitHub Container Registry 푸시
   - 보안 스캔 실행

2. **스테이징 배포**
   - Kubernetes 스테이징 환경 배포
   - 헬스체크 및 검증
   - 롤백 준비

3. **프로덕션 배포**
   - 스테이징 성공 후 프로덕션 배포
   - 무중단 배포 (Rolling Update)
   - 모니터링 및 알림

## 🔍 모니터링 및 알림

### **상태 확인**
- GitHub Actions 페이지에서 실시간 상태 확인
- 각 단계별 로그 및 결과 확인
- 실패 시 자동 롤백 가이드 제공

### **알림**
- 배포 성공/실패 알림
- 보안 취약점 발견 시 알림
- 품질 기준 미달 시 알림

## 🛠️ 문제 해결

### **일반적인 문제들**
1. **Docker 빌드 실패**
   - 캐시 클리어 후 재시도
   - 빌드 컨텍스트 확인

2. **배포 실패**
   - Kubernetes 클러스터 상태 확인
   - 이미지 태그 및 경로 확인
   - 롤백 스크립트 실행

3. **테스트 실패**
   - 로컬에서 테스트 실행
   - 의존성 버전 호환성 확인

### **디버깅 팁**
- GitHub Actions 로그 상세 분석
- 각 단계별 아티팩트 확인
- 환경 변수 및 시크릿 설정 확인

## 📈 성능 최적화

### **빌드 시간 단축**
- Docker 레이어 캐시 활용
- 병렬 작업 실행
- 불필요한 파일 제외 (.dockerignore)

### **리소스 최적화**
- 적절한 runner 선택
- 캐시 전략 최적화
- 이미지 크기 최소화

## 🔐 보안 고려사항

### **시크릿 관리**
- Kubernetes 인증 정보
- 데이터베이스 연결 정보
- API 키 및 토큰

### **접근 제어**
- 환경별 권한 분리
- PR 기반 검토 프로세스
- 자동화된 보안 검사

## 📚 추가 리소스

- [GitHub Actions 공식 문서](https://docs.github.com/en/actions)
- [Docker Buildx 가이드](https://docs.docker.com/buildx/)
- [Kubernetes 배포 가이드](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [Trivy 보안 스캔](https://aquasecurity.github.io/trivy/)
