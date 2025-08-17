# CD 워크플로우 가이드

## 개요

CD(Continuous Deployment) 워크플로우는 CI가 성공적으로 완료된 후 Docker 이미지를 빌드하고 GitHub Container Registry에 푸시하는 자동화 파이프라인입니다.

## 워크플로우 파일

- **위치**: `.github/workflows/cd.yml`
- **트리거**: CI 워크플로우 완료 후 자동 실행
- **수동 실행**: GitHub Actions에서 `workflow_dispatch`로 수동 실행 가능

## 실행 조건

1. **CI 성공**: CI 워크플로우가 성공적으로 완료되어야 함
2. **브랜치**: `main` 브랜치에 푸시된 경우
3. **권한**: GitHub Container Registry에 푸시할 수 있는 권한 필요

## 주요 단계

### 1. CI 결과 확인
- CI 워크플로우의 성공 여부 확인
- 실패 시 Docker 빌드 건너뜀

### 2. 사전 검사
- Dockerfile 존재 확인
- requirements.txt 존재 확인
- src 디렉토리 존재 확인

### 3. Docker 빌드
- Python 3.11 slim 이미지 기반
- 멀티 스테이지 빌드로 최적화
- GitHub Actions 캐시 활용

### 4. 이미지 푸시
- GitHub Container Registry (ghcr.io)에 푸시
- 태그: `latest`, `{branch}-{sha}`, `{sha}`

### 5. 보안 스캔
- Trivy를 사용한 컨테이너 보안 취약점 검사
- SARIF 형식으로 결과 출력

## 필요한 설정

### GitHub Secrets

```bash
PAT_TOKEN: GitHub Personal Access Token
```

### 권한 설정

1. **Repository Settings** → **Actions** → **General**
2. **Workflow permissions** → **Read and write permissions** 선택
3. **Allow GitHub Actions to create and approve pull requests** 체크

### Package 권한

1. **Repository Settings** → **Packages**
2. **Actions access** → **Allow GitHub Actions to create packages** 선택

## 로컬 테스트

### Docker 빌드 테스트

```bash
# 이미지 빌드
docker build -t attention-mil:test .

# 컨테이너 실행
docker run -d -p 8000:8000 --name attention-mil-test attention-mil:test

# API 테스트
curl http://localhost:8000/health/

# 컨테이너 정리
docker stop attention-mil-test
docker rm attention-mil-test
```

### 이미지 검사

```bash
# 이미지 정보 확인
docker inspect attention-mil:test

# 이미지 크기 확인
docker images attention-mil:test

# 이미지 히스토리 확인
docker history attention-mil:test
```

## 문제 해결

### 일반적인 오류

1. **Dockerfile 문법 오류**
   - Dockerfile 문법 검사
   - 베이스 이미지 존재 여부 확인

2. **의존성 설치 실패**
   - requirements.txt 형식 확인
   - 패키지 호환성 검사

3. **권한 오류**
   - PAT_TOKEN 설정 확인
   - Repository 권한 확인

4. **빌드 컨텍스트 문제**
   - .dockerignore 파일 확인
   - 필요한 파일들이 포함되어 있는지 확인

### 로그 확인

```bash
# GitHub Actions 로그
# Actions 탭에서 워크플로우 실행 로그 확인

# Docker 빌드 로그
docker build --progress=plain -t attention-mil:test .
```

## 성능 최적화

### 빌드 캐시

- GitHub Actions 캐시 활용
- Docker 레이어 캐싱
- 의존성 설치 최적화

### 이미지 크기

- 멀티 스테이지 빌드
- 불필요한 파일 제거
- 베이스 이미지 최적화

## 모니터링

### 빌드 상태

- GitHub Actions 대시보드
- 이메일 알림 설정
- Slack/Teams 연동

### 보안 모니터링

- Trivy 스캔 결과
- GitHub Security Advisories
- 의존성 취약점 알림

## 배포 전략

### 태그 전략

- `latest`: 최신 안정 버전
- `{branch}-{sha}`: 브랜치별 빌드
- `{sha}`: 커밋별 빌드

### 롤백 전략

- 이전 버전 이미지 유지
- 태그 기반 버전 관리
- 빠른 롤백 가능

## 참고 자료

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [Trivy Security Scanner](https://aquasecurity.github.io/trivy/)
