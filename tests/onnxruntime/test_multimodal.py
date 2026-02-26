from typing import Optional

import requests
import torch
from PIL import Image
from onnxruntime import InferenceSession
from optimum.utils import is_transformers_version
from optimum.utils.testing_utils import grid_parameters
from parameterized import parameterized
from transformers import AutoProcessor, AutoModelForImageTextToText, DynamicCache, \
    EncoderDecoderCache, PretrainedConfig, set_seed
from transformers.cache_utils import Cache

from optimum.onnxruntime.modeling_multimodal import ORTModelForImageTextToText
from tests.onnxruntime.testing_utils import ORTModelTestMixin, MODEL_NAMES, SEED


class ORTModelForImageTextToTextIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = ["gemma3"]  # noqa: RUF012

    ORTMODEL_CLASS = ORTModelForImageTextToText
    AUTOMODEL_CLASS = AutoModelForImageTextToText
    TRUST_REMOTE_CODE_MODELS = []  # noqa: RUF012

    MODEL_ATOL = {  # noqa: RUF012
        "gemma3": 0.15,
    }
    MODEL_RTOL = {  # noqa: RUF012
        "gemma3": 0.1,
    }

    TASK = "image-text-to-text"

    def _get_sample_image(self):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        return image

    def get_processor(self, model_arch: str):
        return AutoProcessor.from_pretrained(MODEL_NAMES[model_arch])

    def get_inputs(self, model_arch: str):
        processor = self.get_processor(model_arch)
        image = self._get_sample_image()
        text = "<start_of_image>What is in this image?"
        inputs = processor(text=text, images=image, return_tensors="pt")
        return inputs

    def get_transformers_model(self, model_arch: str, trust_remote_code: bool = False, **kwargs):
        model_kwargs = {}
        if trust_remote_code:
            model_kwargs["trust_remote_code"] = True
        set_seed(SEED)
        model = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], **model_kwargs).eval()
        return model

    def get_onnx_model(self, test_name: str, trust_remote_code: bool = False, use_cache: bool = True, **kwargs):
        onnx_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name],
            use_cache=use_cache,
            trust_remote_code=trust_remote_code,
        )
        return onnx_model

    def check_onnx_model_attributes(self, onnx_model, use_cache: bool = True):
        self.assertIsInstance(onnx_model, self.ORTMODEL_CLASS)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForImageTextToText.from_pretrained(MODEL_NAMES["bert"], export=True)
        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [False]}))
    def test_compare_logits_to_transformers(self, test_name: str, model_arch: str, use_cache: bool):
        trust_remote_code = model_arch in self.TRUST_REMOTE_CODE_MODELS
        setup_args = {
            "test_name": test_name,
            "use_cache": use_cache,
            "model_arch": model_arch,
            "trust_remote_code": trust_remote_code,
        }
        self._setup(setup_args)

        model = self.get_transformers_model(**setup_args)
        onnx_model = self.get_onnx_model(**setup_args)
        self.check_onnx_model_attributes(onnx_model, use_cache=use_cache)

        inputs = self.get_inputs(model_arch)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=use_cache)
        onnx_outputs = onnx_model(**inputs, use_cache=use_cache)
        self.compare_logits(inputs, outputs, onnx_outputs, onnx_model=onnx_model, model_arch=model_arch, use_cache=use_cache)

    def compare_logits(
        self,
        inputs,
        outputs1,
        outputs2,
        onnx_model: ORTModelForImageTextToText,
        model_arch: str,
        use_cache: Optional[bool] = None,
    ):
        self.assertTrue("logits" in outputs1)
        self.assertTrue("logits" in outputs2)
        self.assertIsInstance(outputs1.logits, torch.Tensor)
        self.assertIsInstance(outputs2.logits, torch.Tensor)

        atol = self.MODEL_ATOL.get(model_arch, self.ATOL)
        rtol = self.MODEL_RTOL.get(model_arch, self.RTOL)

        torch.testing.assert_close(outputs1.logits, outputs2.logits, atol=atol, rtol=rtol)

        if use_cache:
            self.assertTrue("past_key_values" in outputs1)
            self.assertTrue("past_key_values" in outputs2)
            self.assertIsInstance(outputs1.past_key_values, (tuple, list, Cache))
            self.assertIsInstance(outputs2.past_key_values, (tuple, list, Cache))

            if isinstance(outputs1.past_key_values, DynamicCache):
                if hasattr(outputs1.past_key_values, "to_legacy_cache"):
                    outputs1.past_key_values = outputs1.past_key_values.to_legacy_cache()
                else:
                    outputs1.past_key_values = [
                        (layer.keys, layer.values) for layer in outputs1.past_key_values.layers
                    ]
            elif is_transformers_version(">=", "4.54") and isinstance(outputs1.past_key_values, EncoderDecoderCache):
                # error in latest transformers versions where GPTBigCode returns an EncoderDecoderCache
                if hasattr(outputs1.past_key_values.self_attention_cache, "to_legacy_cache"):
                    outputs1.past_key_values = outputs1.past_key_values.self_attention_cache.to_legacy_cache()
                else:
                    outputs1.past_key_values = [
                        (layer.keys, layer.values) for layer in outputs1.past_key_values.self_attention_cache.layers
                    ]

            if isinstance(outputs2.past_key_values, DynamicCache):
                if hasattr(outputs2.past_key_values, "to_legacy_cache"):
                    outputs2.past_key_values = outputs2.past_key_values.to_legacy_cache()
                else:
                    outputs2.past_key_values = [
                        (layer.keys, layer.values) for layer in outputs2.past_key_values.layers
                    ]
            elif is_transformers_version(">=", "4.54") and isinstance(outputs2.past_key_values, EncoderDecoderCache):
                # error in latest transformers versions where GPTBigCode returns an EncoderDecoderCache
                if hasattr(outputs2.past_key_values.self_attention_cache, "to_legacy_cache"):
                    outputs2.past_key_values = outputs2.past_key_values.self_attention_cache.to_legacy_cache()
                else:
                    outputs2.past_key_values = [
                        (layer.keys, layer.values) for layer in outputs2.past_key_values.self_attention_cache.layers
                    ]

            if is_transformers_version("<", "4.39.0") and "attention_mask" in inputs:
                self.mask_past_key_values(onnx_model, outputs1.past_key_values, inputs["attention_mask"])
                self.mask_past_key_values(onnx_model, outputs2.past_key_values, inputs["attention_mask"])

            torch.testing.assert_close(
                outputs1.past_key_values, outputs2.past_key_values, atol=self.ATOL, rtol=self.RTOL
            )
