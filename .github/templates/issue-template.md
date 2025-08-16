# 🐛 이슈 보고

## 📋 이슈 유형

- [ ] 🐛 버그 리포트
- [ ] ✨ 기능 요청
- [ ] 📚 문서 개선
- [ ] 🧪 테스트 관련
- [ ] 🚀 성능 개선
- [ ] 🔒 보안 이슈
- [ ] 🐳 Docker 관련
- [ ] ☸️ Kubernetes 관련
- [ ] 📊 MLflow 관련
- [ ] 🔄 CI/CD 관련

## 📝 이슈 제목

간단하고 명확한 제목을 작성해주세요.

## 📋 설명

이슈에 대한 자세한 설명을 작성해주세요.

### 현재 상황
현재 어떤 상황인지 설명해주세요.

### 예상 동작
어떤 동작을 기대했는지 설명해주세요.

### 실제 동작
실제로는 어떤 동작이 발생했는지 설명해주세요.

## 🔍 재현 방법

버그를 재현할 수 있는 단계별 방법을 작성해주세요.

```bash
# 1. 환경 설정
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. 실행
python scripts/run_api.py

# 3. 테스트
curl -X POST http://localhost:8000/predict/ \
  -F "file=@test_image.jpg"
```

### 환경 정보
- **OS**: Ubuntu 20.04 / Windows 10 / macOS 12.0
- **Python 버전**: 3.9.7
- **PyTorch 버전**: 2.0.0
- **CUDA 버전**: 11.8 (GPU 사용 시)

## 📊 로그 및 에러 메시지

### 에러 로그
```
Traceback (most recent call last):
  File "scripts/run_api.py", line 45, in <module>
    main()
  File "scripts/run_api.py", line 32, in main
    app.run(host=args.host, port=args.port)
TypeError: 'FastAPI' object has no attribute 'run'
```

### 시스템 로그
```
2024-12-01 10:30:15 - ERROR - Failed to load model
2024-12-01 10:30:16 - ERROR - Model file not found: models/attention_mil.onnx
```

## 🖼️ 스크린샷

문제 상황을 보여주는 스크린샷이 있다면 첨부해주세요.

## 💡 해결 방안

문제 해결을 위한 아이디어가 있다면 작성해주세요.

### 제안하는 해결 방법
1. **방법 1**: FastAPI 대신 uvicorn 사용
2. **방법 2**: 모델 파일 경로 확인
3. **방법 3**: 의존성 버전 호환성 검사

### 대안
- **대안 1**: 다른 모델 로딩 방식 사용
- **대안 2**: 환경 변수 설정 확인
- **대안 3**: Docker 컨테이너 사용

## 🔗 관련 정보

### 관련 파일
- `src/api/main.py`
- `scripts/run_api.py`
- `requirements.txt`

### 관련 이슈
- #123 (유사한 문제)
- #456 (연관된 기능)

### 관련 PR
- #789 (해결 시도)

## 📅 우선순위

- [ ] 🔴 높음 (긴급)
- [ ] 🟡 보통 (중요)
- [ ] 🟢 낮음 (개선)

## 👥 담당자

- [ ] @mlops-team
- [ ] @backend-team
- [ ] @ai-team
- [ ] @devops-team

## 🏷️ 라벨

적절한 라벨을 선택해주세요:

- `bug` - 버그 리포트
- `enhancement` - 기능 요청
- `documentation` - 문서 관련
- `testing` - 테스트 관련
- `performance` - 성능 관련
- `security` - 보안 관련
- `docker` - Docker 관련
- `kubernetes` - Kubernetes 관련
- `mlflow` - MLflow 관련
- `ci-cd` - CI/CD 관련
- `help wanted` - 도움 요청
- `good first issue` - 초보자용

## 📝 추가 정보

추가적인 정보나 참고 자료가 있다면 여기에 작성해주세요.

### 환경 설정 파일
```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/attention_mil.onnx
```

### 설정 파일
```yaml
# configs/model_configs/attention_mil.yaml
model:
  type: attention_mil
  feature_dim: 1408
  num_classes: 3
  attention_dim: 512
```

---

**이슈 번호**: #(자동 생성)  
**작성자**: @(GitHub 사용자명)  
**작성일**: $(date)  
**상태**: 🆕 신규 