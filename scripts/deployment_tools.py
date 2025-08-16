#!/usr/bin/env python3
"""
배포 도구 통합 스크립트

Docker 빌드, Kubernetes 배포, 모델 검증 등 배포 관련 모든 작업을 수행합니다.
"""

import argparse
import os
import sys
import subprocess
import yaml
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import shutil

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from configs import AppConfig

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DockerManager:
    """Docker 이미지 빌드 및 관리 클래스"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.project_root = Path(__file__).parent.parent
        
    def build_image(self, tag: str = None, dockerfile: str = None, 
                   build_args: Dict[str, str] = None) -> bool:
        """Docker 이미지 빌드"""
        try:
            tag = tag or f"attention-mil:{time.strftime('%Y%m%d-%H%M%S')}"
            dockerfile = dockerfile or "Dockerfile"
            build_args = build_args or {}
            
            logger.info(f"🐳 Docker 이미지 빌드 시작: {tag}")
            
            # 기본 빌드 인수
            default_args = {
                'PROJECT_NAME': 'attention-mil',
                'PYTHON_VERSION': '3.9',
                'CUDA_VERSION': '11.8'
            }
            default_args.update(build_args)
            
            # docker build 명령어 구성
            cmd = [
                'docker', 'build',
                '-t', tag,
                '-f', dockerfile,
                '.'
            ]
            
            # 빌드 인수 추가
            for key, value in default_args.items():
                cmd.extend(['--build-arg', f'{key}={value}'])
            
            logger.info(f"실행 명령어: {' '.join(cmd)}")
            
            # 빌드 실행
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Docker 이미지 빌드 완료: {tag}")
                return True
            else:
                logger.error(f"❌ Docker 빌드 실패: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Docker 빌드 중 오류: {e}")
            return False
    
    def push_image(self, tag: str, registry: str = None) -> bool:
        """Docker 이미지 푸시"""
        try:
            if registry:
                full_tag = f"{registry}/{tag}"
                # 이미지 태그 변경
                subprocess.run(['docker', 'tag', tag, full_tag], check=True)
                tag = full_tag
            
            logger.info(f"📤 Docker 이미지 푸시 시작: {tag}")
            
            result = subprocess.run(['docker', 'push', tag], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Docker 이미지 푸시 완료: {tag}")
                return True
            else:
                logger.error(f"❌ Docker 푸시 실패: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Docker 푸시 중 오류: {e}")
            return False
    
    def run_container(self, image: str, name: str = None, 
                     ports: Dict[str, str] = None, env_vars: Dict[str, str] = None) -> bool:
        """Docker 컨테이너 실행"""
        try:
            name = name or f"attention-mil-{int(time.time())}"
            ports = ports or {'8000': '8000'}
            env_vars = env_vars or {}
            
            logger.info(f"🚀 Docker 컨테이너 실행 시작: {name}")
            
            cmd = ['docker', 'run', '-d', '--name', name]
            
            # 포트 매핑
            for host_port, container_port in ports.items():
                cmd.extend(['-p', f'{host_port}:{container_port}'])
            
            # 환경 변수
            for key, value in env_vars.items():
                cmd.extend(['-e', f'{key}={value}'])
            
            cmd.extend([image])
            
            logger.info(f"실행 명령어: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Docker 컨테이너 실행 완료: {name}")
                return True
            else:
                logger.error(f"❌ Docker 컨테이너 실행 실패: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Docker 컨테이너 실행 중 오류: {e}")
            return False


class KubernetesManager:
    """Kubernetes 배포 관리 클래스"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.k8s_dir = Path(__file__).parent.parent / "k8s"
        
    def deploy_to_namespace(self, namespace: str = "dev", 
                           config_files: List[str] = None) -> bool:
        """Kubernetes 네임스페이스에 배포"""
        try:
            if not config_files:
                config_files = ['namespace.yaml', 'configmap.yaml', 'secret.yaml', 'deployment.yaml']
            
            logger.info(f"☸️ Kubernetes 배포 시작: {namespace}")
            
            # 네임스페이스 생성
            namespace_file = self.k8s_dir / namespace / "namespace.yaml"
            if namespace_file.exists():
                subprocess.run(['kubectl', 'apply', '-f', str(namespace_file)], check=True)
                logger.info(f"네임스페이스 생성: {namespace}")
            
            # 설정 파일들 배포
            for config_file in config_files:
                file_path = self.k8s_dir / namespace / config_file
                if file_path.exists():
                    subprocess.run(['kubectl', 'apply', '-f', str(file_path)], check=True)
                    logger.info(f"설정 파일 배포: {config_file}")
                else:
                    logger.warning(f"설정 파일을 찾을 수 없음: {config_file}")
            
            logger.info(f"✅ Kubernetes 배포 완료: {namespace}")
            return True
            
        except Exception as e:
            logger.error(f"Kubernetes 배포 실패: {e}")
            return False
    
    def check_deployment_status(self, namespace: str = "dev") -> Dict[str, Any]:
        """배포 상태 확인"""
        try:
            logger.info(f"🔍 배포 상태 확인: {namespace}")
            
            # 파드 상태 확인
            result = subprocess.run(
                ['kubectl', 'get', 'pods', '-n', namespace, '-o', 'json'],
                capture_output=True, text=True, check=True
            )
            
            pods_info = json.loads(result.stdout)
            
            # 서비스 상태 확인
            result = subprocess.run(
                ['kubectl', 'get', 'services', '-n', namespace, '-o', 'json'],
                capture_output=True, text=True, check=True
            )
            
            services_info = json.loads(result.stdout)
            
            return {
                'namespace': namespace,
                'pods': pods_info,
                'services': services_info,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"배포 상태 확인 실패: {e}")
            return {}
    
    def scale_deployment(self, namespace: str, deployment: str, replicas: int) -> bool:
        """배포 스케일링"""
        try:
            logger.info(f"📈 배포 스케일링: {deployment} -> {replicas} replicas")
            
            subprocess.run([
                'kubectl', 'scale', 'deployment', deployment,
                f'--replicas={replicas}', '-n', namespace
            ], check=True)
            
            logger.info(f"✅ 배포 스케일링 완료: {deployment}")
            return True
            
        except Exception as e:
            logger.error(f"배포 스케일링 실패: {e}")
            return False


class ModelValidator:
    """모델 검증 및 성능 측정 클래스"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        
    def validate_model_performance(self, model_path: str, 
                                 test_data_path: str = None) -> Dict[str, Any]:
        """모델 성능 검증"""
        try:
            logger.info(f"🔍 모델 성능 검증 시작: {model_path}")
            
            # 모델 로드
            import torch
            from src.models.attention_mil import AttentionMIL
            
            model = AttentionMIL(
                num_classes=self.config.attention_mil.num_classes,
                feature_extractor_config=self.config.attention_mil.feature_extractor,
                attention_config=self.config.attention_mil.attention
            )
            
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            
            # 모델 크기 계산
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # 추론 시간 측정
            dummy_input = torch.randn(1, 3, 224, 224)
            
            # GPU 사용 가능 여부 확인
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            dummy_input = dummy_input.to(device)
            
            # 워밍업
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)
            
            # 추론 시간 측정
            start_time = time.time()
            with torch.no_grad():
                for _ in range(100):
                    _ = model(dummy_input)
            end_time = time.time()
            
            avg_inference_time = (end_time - start_time) / 100
            
            # 메모리 사용량 측정
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                memory_reserved = torch.cuda.memory_reserved() / 1024**2    # MB
            else:
                memory_allocated = 0
                memory_reserved = 0
            
            results = {
                'model_path': model_path,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': os.path.getsize(model_path) / 1024**2,
                'avg_inference_time_ms': avg_inference_time * 1000,
                'device': str(device),
                'memory_allocated_mb': memory_allocated,
                'memory_reserved_mb': memory_reserved,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info("✅ 모델 성능 검증 완료")
            return results
            
        except Exception as e:
            logger.error(f"모델 성능 검증 실패: {e}")
            return {}
    
    def compare_models(self, model_paths: List[str]) -> Dict[str, Any]:
        """여러 모델 성능 비교"""
        logger.info("🔍 모델 성능 비교 시작")
        
        comparison_results = {}
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                result = self.validate_model_performance(model_path)
                if result:
                    comparison_results[os.path.basename(model_path)] = result
        
        # 성능 순위 결정
        if comparison_results:
            sorted_models = sorted(
                comparison_results.items(),
                key=lambda x: x[1]['avg_inference_time_ms']
            )
            
            comparison_results['ranking'] = [
                {'model': name, 'inference_time': data['avg_inference_time_ms']}
                for name, data in sorted_models
            ]
        
        return comparison_results


class DeploymentOrchestrator:
    """전체 배포 프로세스 조율 클래스"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.docker_manager = DockerManager(config)
        self.k8s_manager = KubernetesManager(config)
        self.model_validator = ModelValidator(config)
        
    def full_deployment_pipeline(self, model_path: str, 
                                image_tag: str = None,
                                namespace: str = "dev") -> bool:
        """전체 배포 파이프라인 실행"""
        try:
            logger.info("🚀 전체 배포 파이프라인 시작")
            
            # 1. 모델 검증
            logger.info("1️⃣ 모델 성능 검증")
            validation_result = self.model_validator.validate_model_performance(model_path)
            if not validation_result:
                logger.error("모델 검증 실패")
                return False
            
            # 2. Docker 이미지 빌드
            logger.info("2️⃣ Docker 이미지 빌드")
            if not self.docker_manager.build_image(tag=image_tag):
                logger.error("Docker 이미지 빌드 실패")
                return False
            
            # 3. Kubernetes 배포
            logger.info("3️⃣ Kubernetes 배포")
            if not self.k8s_manager.deploy_to_namespace(namespace):
                logger.error("Kubernetes 배포 실패")
                return False
            
            # 4. 배포 상태 확인
            logger.info("4️⃣ 배포 상태 확인")
            time.sleep(30)  # 배포 완료 대기
            status = self.k8s_manager.check_deployment_status(namespace)
            
            logger.info("✅ 전체 배포 파이프라인 완료")
            return True
            
        except Exception as e:
            logger.error(f"배포 파이프라인 실패: {e}")
            return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="배포 도구 통합 스크립트")
    parser.add_argument("--action", required=True,
                       choices=["docker-build", "docker-push", "docker-run", "k8s-deploy", 
                               "k8s-status", "k8s-scale", "validate-model", "compare-models", 
                               "full-pipeline"],
                       help="수행할 작업")
    parser.add_argument("--config", default="configs/attention_mil.yaml",
                       help="설정 파일 경로")
    parser.add_argument("--tag", help="Docker 이미지 태그")
    parser.add_argument("--namespace", default="dev", help="Kubernetes 네임스페이스")
    parser.add_argument("--model-path", help="모델 파일 경로")
    parser.add_argument("--model-paths", nargs='+', help="비교할 모델 파일 경로들")
    parser.add_argument("--replicas", type=int, help="배포 복제본 수")
    parser.add_argument("--deployment", help="배포 이름")
    
    args = parser.parse_args()
    
    # 설정 로드
    try:
        config = AppConfig.from_yaml(args.config)
    except Exception as e:
        logger.error(f"설정 로드 실패: {e}")
        return
    
    # 오케스트레이터 생성
    orchestrator = DeploymentOrchestrator(config)
    
    if args.action == "docker-build":
        # Docker 이미지 빌드
        orchestrator.docker_manager.build_image(tag=args.tag)
    
    elif args.action == "docker-push":
        # Docker 이미지 푸시
        if not args.tag:
            logger.error("--tag 인수가 필요합니다")
            return
        orchestrator.docker_manager.push_image(args.tag)
    
    elif args.action == "docker-run":
        # Docker 컨테이너 실행
        if not args.tag:
            logger.error("--tag 인수가 필요합니다")
            return
        orchestrator.docker_manager.run_container(args.tag)
    
    elif args.action == "k8s-deploy":
        # Kubernetes 배포
        orchestrator.k8s_manager.deploy_to_namespace(args.namespace)
    
    elif args.action == "k8s-status":
        # Kubernetes 배포 상태 확인
        status = orchestrator.k8s_manager.check_deployment_status(args.namespace)
        print(json.dumps(status, indent=2, ensure_ascii=False))
    
    elif args.action == "k8s-scale":
        # Kubernetes 배포 스케일링
        if not args.deployment or not args.replicas:
            logger.error("--deployment와 --replicas 인수가 필요합니다")
            return
        orchestrator.k8s_manager.scale_deployment(args.namespace, args.deployment, args.replicas)
    
    elif args.action == "validate-model":
        # 모델 검증
        if not args.model_path:
            logger.error("--model-path 인수가 필요합니다")
            return
        result = orchestrator.model_validator.validate_model_performance(args.model_path)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif args.action == "compare-models":
        # 모델 비교
        if not args.model_paths:
            logger.error("--model-paths 인수가 필요합니다")
            return
        result = orchestrator.model_validator.compare_models(args.model_paths)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif args.action == "full-pipeline":
        # 전체 배포 파이프라인
        if not args.model_path:
            logger.error("--model-path 인수가 필요합니다")
            return
        orchestrator.full_deployment_pipeline(args.model_path, args.tag, args.namespace)


if __name__ == "__main__":
    main()

