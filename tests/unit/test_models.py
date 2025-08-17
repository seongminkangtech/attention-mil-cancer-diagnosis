"""
모델 단위 테스트

의료 AI 모델들의 동작을 테스트합니다.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from src.models.attention_mil import AttentionMILModel
from src.models.feature_extractor import FeatureExtractor
from src.models.onnx_model import ONNXModel


class TestFeatureExtractor:
    """FeatureExtractor 클래스 테스트"""

    def test_feature_extractor_initialization(self):
        """FeatureExtractor 초기화 테스트"""
        # 실제 생성자 매개변수에 맞게 수정
        extractor = FeatureExtractor(model_name="efficientnet_b2")

        assert extractor is not None
        assert extractor.feature_dim > 0

    def test_feature_extractor_output_shape(self):
        """특징 추출기 출력 형태 테스트"""
        extractor = FeatureExtractor(model_name="efficientnet_b2")

        # 테스트 입력 생성
        batch_size = 2
        channels = 3
        height = 256
        width = 256

        test_input = torch.randn(batch_size, channels, height, width)

        with torch.no_grad():
            output = extractor(test_input)

        assert output.shape[0] == batch_size
        assert output.shape[1] == extractor.feature_dim

    def test_feature_extractor_invalid_backbone(self):
        """잘못된 백본 모델명 테스트"""
        with pytest.raises(Exception):  # timm에서 오류 발생
            FeatureExtractor(model_name="invalid_model")


class TestAttentionMILModel:
    """AttentionMILModel 클래스 테스트"""

    def test_attention_mil_initialization(self):
        """AttentionMILModel 초기화 테스트"""
        # 실제 생성자 매개변수에 맞게 수정
        feature_extractor = FeatureExtractor(model_name="efficientnet_b2")
        model = AttentionMILModel(
            num_classes=3, feature_extractor=feature_extractor, attention_hidden_dim=128
        )

        assert model is not None
        assert model.num_classes == 3

    def test_attention_mil_forward(self):
        """AttentionMILModel 순전파 테스트"""
        feature_extractor = FeatureExtractor(model_name="efficientnet_b2")
        model = AttentionMILModel(
            num_classes=3, feature_extractor=feature_extractor, attention_hidden_dim=128
        )

        # 테스트 입력 생성 (B, num_tiles, C, H, W)
        batch_size = 2
        num_tiles = 10
        channels = 3
        height = 256
        width = 256

        test_input = torch.randn(batch_size, num_tiles, channels, height, width)

        with torch.no_grad():
            output = model(test_input)

        assert output.shape == (batch_size, 3)

    def test_attention_mil_attention_weights(self):
        """Attention 가중치 계산 테스트"""
        feature_extractor = FeatureExtractor(model_name="efficientnet_b2")
        model = AttentionMILModel(
            num_classes=3, feature_extractor=feature_extractor, attention_hidden_dim=128
        )

        batch_size = 1
        num_tiles = 5
        channels = 3
        height = 256
        width = 256

        test_input = torch.randn(batch_size, num_tiles, channels, height, width)

        with torch.no_grad():
            attention_weights = model.get_attention_weights(test_input)

        assert attention_weights.shape == (batch_size, num_tiles)
        # Attention 가중치 합이 1에 가까워야 함
        assert torch.allclose(
            torch.sum(attention_weights, dim=1), torch.ones(batch_size), atol=1e-6
        )

    def test_attention_mil_invalid_input(self):
        """잘못된 입력 형태 테스트"""
        feature_extractor = FeatureExtractor(model_name="efficientnet_b2")
        model = AttentionMILModel(
            num_classes=3, feature_extractor=feature_extractor, attention_hidden_dim=128
        )

        # 잘못된 차원의 입력 (5차원이어야 하는데 4차원)
        invalid_input = torch.randn(
            2, 3, 256, 256
        )  # (B, C, H, W) - num_tiles 차원 누락

        with pytest.raises(ValueError):
            with torch.no_grad():
                model(invalid_input)


class TestONNXModel:
    """ONNXModel 클래스 테스트"""

    def test_onnx_model_initialization(self):
        """ONNXModel 초기화 테스트"""
        # Mock ONNX 모델 파일 경로
        mock_model_path = "mock_model.onnx"

        # 파일 존재 여부를 Mock으로 처리
        with patch("os.path.exists", return_value=True):
            with patch("onnxruntime.InferenceSession") as mock_session:
                # Mock 세션 인스턴스 생성
                mock_session_instance = Mock()

                # get_inputs()와 get_outputs() 메서드 Mock
                mock_input = Mock()
                mock_input.name = "input"
                mock_input.shape = [1, 10, 3, 256, 256]
                mock_input.type = "tensor(float)"

                mock_output = Mock()
                mock_output.name = "output"
                mock_output.shape = [1, 3]
                mock_output.type = "tensor(float)"

                mock_session_instance.get_inputs.return_value = [mock_input]
                mock_session_instance.get_outputs.return_value = [mock_output]

                mock_session.return_value = mock_session_instance

                model = ONNXModel(mock_model_path)

                assert model is not None
                assert model.model_path == mock_model_path

    def test_onnx_model_predict(self):
        """ONNX 모델 예측 테스트"""
        mock_model_path = "mock_model.onnx"

        with patch("os.path.exists", return_value=True):
            with patch("onnxruntime.InferenceSession") as mock_session:
                # Mock 세션 설정
                mock_session_instance = Mock()

                # get_inputs()와 get_outputs() 메서드 Mock
                mock_input = Mock()
                mock_input.name = "input"
                mock_input.shape = [1, 10, 3, 256, 256]
                mock_input.type = "tensor(float)"

                mock_output = Mock()
                mock_output.name = "output"
                mock_output.shape = [1, 3]
                mock_output.type = "tensor(float)"

                mock_session_instance.get_inputs.return_value = [mock_input]
                mock_session_instance.get_outputs.return_value = [mock_output]

                # run() 메서드가 올바른 반환값을 제공하도록 Mock
                mock_session_instance.run.return_value = [
                    np.random.rand(1, 3),
                    np.random.rand(1, 10),
                ]

                mock_session.return_value = mock_session_instance

                model = ONNXModel(mock_model_path)

                # 테스트 입력
                test_input = np.random.rand(1, 10, 3, 256, 256).astype(np.float32)

                # 예측 수행
                prediction, attention_weights = model.predict(test_input)

                assert prediction is not None
                assert attention_weights is not None

    def test_onnx_model_invalid_input_type(self):
        """잘못된 입력 타입 테스트"""
        mock_model_path = "mock_model.onnx"

        with patch("os.path.exists", return_value=True):
            with patch("onnxruntime.InferenceSession") as mock_session:
                # Mock 세션 설정
                mock_session_instance = Mock()

                mock_input = Mock()
                mock_input.name = "input"
                mock_input.shape = [1, 10, 3, 256, 256]
                mock_input.type = "tensor(float)"

                mock_output = Mock()
                mock_output.name = "output"
                mock_output.shape = [1, 3]
                mock_output.type = "tensor(float)"

                mock_session_instance.get_inputs.return_value = [mock_input]
                mock_session_instance.get_outputs.return_value = [mock_output]

                mock_session.return_value = mock_session_instance

                model = ONNXModel(mock_model_path)

                # 잘못된 타입의 입력
                invalid_input = "invalid_input"

                with pytest.raises(ValueError):
                    model.predict(invalid_input)

    def test_onnx_model_input_shape_validation(self):
        """입력 형태 검증 테스트"""
        mock_model_path = "mock_model.onnx"

        with patch("os.path.exists", return_value=True):
            with patch("onnxruntime.InferenceSession") as mock_session:
                # Mock 세션 설정
                mock_session_instance = Mock()

                mock_input = Mock()
                mock_input.name = "input"
                mock_input.shape = [1, 10, 3, 256, 256]
                mock_input.type = "tensor(float)"

                mock_output = Mock()
                mock_output.name = "output"
                mock_output.shape = [1, 3]
                mock_output.type = "tensor(float)"

                mock_session_instance.get_inputs.return_value = [mock_input]
                mock_session_instance.get_outputs.return_value = [mock_output]

                mock_session.return_value = mock_session_instance

                model = ONNXModel(mock_model_path)

                # 잘못된 형태의 입력
                invalid_input = np.random.rand(3, 256, 256).astype(
                    np.float32
                )  # 차원 부족

                with pytest.raises(ValueError):
                    model.predict(invalid_input)


class TestModelIntegration:
    """모델 통합 테스트"""

    def test_feature_extractor_to_attention_mil(self):
        """FeatureExtractor와 AttentionMILModel 연동 테스트"""
        # 모델 초기화
        feature_extractor = FeatureExtractor(model_name="efficientnet_b2")
        attention_mil = AttentionMILModel(
            num_classes=3, feature_extractor=feature_extractor, attention_hidden_dim=128
        )

        # 테스트 입력
        batch_size = 1
        num_tiles = 5
        channels = 3
        height = 256
        width = 256

        test_input = torch.randn(batch_size, num_tiles, channels, height, width)

        # 통합 테스트
        with torch.no_grad():
            output = attention_mil(test_input)
            attention_weights = attention_mil.get_attention_weights(test_input)

        # 출력 검증
        assert output.shape == (batch_size, 3)
        assert attention_weights.shape == (batch_size, num_tiles)

        # Attention 가중치 합이 1에 가까워야 함
        assert torch.allclose(
            torch.sum(attention_weights, dim=1), torch.ones(batch_size), atol=1e-6
        )


if __name__ == "__main__":
    pytest.main([__file__])
