#!/usr/bin/env python3
"""
Attention MIL 모델 학습 스크립트

의료 영상 분류를 위한 Attention MIL 모델을 학습합니다.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import time
import datetime
from tqdm import tqdm
import torchmetrics
import mlflow
import mlflow.pytorch

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import AttentionMILModel, FeatureExtractor
from src.utils import load_and_preprocess_images, create_data_loaders, get_class_names


def load_config(config_path: str) -> dict:
    """
    설정 파일 로드
    
    Args:
        config_path (str): 설정 파일 경로
        
    Returns:
        dict: 설정 딕셔너리
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(device_config: str) -> torch.device:
    """
    디바이스 설정
    
    Args:
        device_config (str): 디바이스 설정 ("auto", "cuda", "cpu")
        
    Returns:
        torch.device: 설정된 디바이스
    """
    if device_config == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config)
    
    print(f"사용 디바이스: {device}")
    return device


def create_model(config: dict, device: torch.device) -> AttentionMILModel:
    """
    모델 생성
    
    Args:
        config (dict): 설정 딕셔너리
        device (torch.device): 디바이스
        
    Returns:
        AttentionMILModel: 생성된 모델
    """
    # 특징 추출기 생성
    feature_extractor = FeatureExtractor(
        model_name=config['model']['feature_extractor']['model_name'],
        pretrained=config['model']['feature_extractor']['pretrained']
    )
    
    # Attention MIL 모델 생성
    model = AttentionMILModel(
        num_classes=config['model']['num_classes'],
        feature_extractor=feature_extractor,
        attention_hidden_dim=config['model']['attention']['hidden_dim'],
        dropout_rate=config['model']['attention']['dropout_rate']
    )
    
    model = model.to(device)
    return model


def create_optimizer_and_scheduler(model: nn.Module, config: dict):
    """
    옵티마이저와 스케줄러 생성
    
    Args:
        model (nn.Module): 모델
        config (dict): 설정 딕셔너리
        
    Returns:
        tuple: (옵티마이저, 스케줄러)
    """
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    scheduler = lr_scheduler.StepLR(
        optimizer, 
        step_size=config['training']['step_size'], 
        gamma=config['training']['gamma']
    )
    
    return optimizer, scheduler


def train_epoch(model: nn.Module, 
                train_loader, 
                optimizer: optim.Optimizer,
                device: torch.device,
                epoch: int,
                config: dict) -> tuple:
    """
    한 에포크 학습
    
    Args:
        model (nn.Module): 모델
        train_loader: 훈련 데이터 로더
        optimizer (optim.Optimizer): 옵티마이저
        device (torch.device): 디바이스
        epoch (int): 현재 에포크
        config (dict): 설정 딕셔너리
        
    Returns:
        tuple: (평균 손실, 평균 정확도)
    """
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    num_batches = 0
    
    # 정확도 메트릭
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=config['model']['num_classes']).to(device)
    
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
    
    for batch_idx, (ids, images, labels) in enumerate(train_bar):
        images = images.to(device).float()
        labels = labels.to(device).float()
        
        optimizer.zero_grad()
        
        # 순전파
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        
        # 역전파
        loss.backward()
        optimizer.step()
        
        # 메트릭 계산
        acc = accuracy(outputs.softmax(dim=1).argmax(dim=1), labels.argmax(dim=1))
        
        running_loss += loss.item()
        running_acc += acc.item()
        num_batches += 1
        
        # 진행률 업데이트
        train_bar.set_postfix({
            'Loss': f"{running_loss/num_batches:.4f}",
            'Acc': f"{running_acc/num_batches:.4f}"
        })
    
    return running_loss / num_batches, running_acc / num_batches


def validate_epoch(model: nn.Module, 
                  val_loader, 
                  device: torch.device,
                  config: dict) -> tuple:
    """
    검증 에포크
    
    Args:
        model (nn.Module): 모델
        val_loader: 검증 데이터 로더
        device (torch.device): 디바이스
        config (dict): 설정 딕셔너리
        
    Returns:
        tuple: (평균 손실, 평균 정확도)
    """
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    num_batches = 0
    
    # 정확도 메트릭
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=config['model']['num_classes']).to(device)
    
    with torch.no_grad():
        for ids, images, labels in tqdm(val_loader, desc="검증"):
            images = images.to(device).float()
            labels = labels.to(device).float()
            
            # 순전파
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            
            # 메트릭 계산
            acc = accuracy(outputs.softmax(dim=1).argmax(dim=1), labels.argmax(dim=1))
            
            running_loss += loss.item()
            running_acc += acc.item()
            num_batches += 1
    
    return running_loss / num_batches, running_acc / num_batches


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Attention MIL 모델 학습")
    parser.add_argument("--config", type=str, default="configs/model_configs/attention_mil.yaml",
                       help="설정 파일 경로")
    parser.add_argument("--experiment_name", type=str, default="attention_mil_training",
                       help="MLflow 실험 이름")
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 디바이스 설정
    device = setup_device(config['hardware']['device'])
    
    # MLflow 설정
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run():
        # MLflow에 설정 로깅
        mlflow.log_params({
            "learning_rate": config['training']['learning_rate'],
            "batch_size": config['data']['batch_size'],
            "epochs": config['training']['epochs'],
            "model_name": config['model']['feature_extractor']['model_name']
        })
        
        print("데이터 로딩 중...")
        
        # 데이터 로드
        id_list, image_tensor, labels = load_and_preprocess_images(
            data_path=config['paths']['frame_path'],
            csv_file=config['paths']['train_csv'],
            label_file=config['paths']['label_csv'],
            image_count=config['data']['image_count'],
            img_size=config['data']['img_size']
        )
        
        # 데이터 로더 생성
        train_loader, val_loader = create_data_loaders(
            id_list=id_list,
            image_tensor=image_tensor,
            labels=labels,
            batch_size=config['data']['batch_size'],
            train_ratio=config['data']['train_ratio'],
            shuffle=config['data']['shuffle']
        )
        
        print(f"훈련 데이터: {len(train_loader.dataset)} 샘플")
        print(f"검증 데이터: {len(val_loader.dataset)} 샘플")
        
        # 모델 생성
        model = create_model(config, device)
        
        # 옵티마이저와 스케줄러 생성
        optimizer, scheduler = create_optimizer_and_scheduler(model, config)
        
        # 학습 시작
        print("학습 시작...")
        start_time = time.time()
        
        best_val_acc = 0.0
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        for epoch in range(config['training']['epochs']):
            # 훈련
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, device, epoch, config
            )
            
            # 검증
            val_loss, val_acc = validate_epoch(
                model, val_loader, device, config
            )
            
            # 스케줄러 업데이트
            scheduler.step()
            
            # 결과 저장
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # MLflow에 메트릭 로깅
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": scheduler.get_last_lr()[0]
            }, step=epoch)
            
            print(f"Epoch {epoch+1}/{config['training']['epochs']}")
            print(f"  훈련 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  검증 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # 최고 성능 모델 저장 (MLflow에만 저장)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                mlflow.pytorch.log_model(model, "model")
                
        # 학습 완료
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"학습 완료! 총 소요 시간: {training_time:.2f}초")
        print(f"최고 검증 정확도: {best_val_acc:.4f}")
        
        # MLflow에 최종 결과 로깅
        mlflow.log_metrics({
            "best_val_acc": best_val_acc,
            "training_time": training_time
        })


if __name__ == "__main__":
    main() 