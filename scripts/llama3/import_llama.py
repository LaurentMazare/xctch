# python scripts/llama3/import_llama.py --checkpoint /Users/laurent/tmp/llama-3.2/consolidated.00.pth --params /Users/laurent/tmp/llama-3.2/params.json -kv -X -d bf16 --output_name foo.pte
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# Example script for exporting Llama2 to flatbuffer

import argparse
import copy
import json
import logging
import shlex
from enum import Enum
from json import JSONDecodeError
from pathlib import Path
from typing import Callable, List, Optional, Union

import pkg_resources

import torch

from executorch.devtools.etrecord import generate_etrecord

from executorch.examples.models.llama2.llama_transformer import ModelArgs

from executorch.extension.llm.export.builder import DType, LLMEdgeManager

from executorch.extension.llm.export.partitioner_lib import (
    get_coreml_partitioner,
    get_mps_partitioner,
    get_qnn_partitioner,
    get_vulkan_partitioner,
    get_xnnpack_partitioner,
)

from executorch.extension.llm.export.quantizer_lib import (
    get_coreml_quantizer,
    get_pt2e_quantization_params,
    get_pt2e_quantizers,
    get_qnn_quantizer,
)
from executorch.util.activation_memory_profiler import generate_memory_trace

import importlib
import os
from typing import Any, Tuple

import torch

import json
import os
from pathlib import Path

import torch

from executorch.examples.models.llama2.llama_transformer import ModelArgs, Transformer

from abc import ABC, abstractmethod

import torch


class EagerModelBase(ABC):
    """
    Abstract base class for eager mode models.

    This abstract class defines the interface that eager mode model classes should adhere to.
    Eager mode models inherit from this class to ensure consistent behavior and structure.
    """

    @abstractmethod
    def __init__(self):
        """
        Constructor for EagerModelBase.

        This initializer may be overridden in derived classes to provide additional setup if needed.
        """
        pass

    @abstractmethod
    def get_eager_model(self) -> torch.nn.Module:
        """
        Abstract method to return an eager PyTorch model instance.

        Returns:
            nn.Module: An instance of a PyTorch model, suitable for eager execution.
        """
        raise NotImplementedError("get_eager_model")

    @abstractmethod
    def get_example_inputs(self):
        """
        Abstract method to provide example inputs for the model.

        Returns:
            Any: Example inputs that can be used for testing and tracing.
        """
        raise NotImplementedError("get_example_inputs")

class Llama2Model(EagerModelBase):
    def __init__(self, **kwargs):
        import pkg_resources

        # default path to the resource file
        # It currently supports 3 ways of specifying the checkpoint location:
        # 1. Using default path locates in examples/models/llama2/params
        # 2. Passing in the checkpoint path and params via kwargs
        # 3. Using the path from pkg_resources, only works with buck2
        try:
            # The 3rd way, if we can import this path, we are running with buck2, all resources can be accessed with pkg_resources.resource_filename
            # pyre-ignore
            from executorch.examples.models.llama2 import params

            ckpt_dir = Path(
                pkg_resources.resource_filename(
                    "executorch.examples.models.llama2", "params"
                )
            )
        except:
            # The 1st way
            ckpt_dir = Path(__file__).absolute().parent / "params"

        # Check if checkpoint_dir was provided for a sharded checkpoint.
        checkpoint_dir = kwargs.get("checkpoint_dir", None)

        # Use single checkpoint file.
        checkpoint_path = kwargs.get("checkpoint", ckpt_dir / "demo_rand_params.pth")

        params_path = kwargs.get("params", ckpt_dir / "demo_config.json")

        self.use_kv_cache = kwargs.get("use_kv_cache", False)
        self.use_sdpa_with_kv_cache_op = kwargs.get("use_sdpa_with_kv_cache", False)
        self.generate_full_logits = kwargs.get("generate_full_logits", False)
        self.enable_dynamic_shape = kwargs.get("enable_dynamic_shape", False)

        self.max_seq_len = kwargs.get("max_seq_len", 128)
        self.args = kwargs.get("args", None)
        # The example is using a dummy small model with random weights for demo purpose only.
        # Follow the instruction in https://github.com/facebookresearch/llama to download the model
        device = "cpu"
        # flake8: noqa: TOR102
        cps = []
        if checkpoint_dir is not None:
            # Load multiple checkpoint; ignore the single path.
            checkpoint_path = None
            for i in range(4):
                cp_name = f"consolidated.{i}.pth"
                print(f"Loading {cp_name}")
                cps.append(
                    torch.load(
                        os.path.join(checkpoint_dir, cp_name),
                        map_location=device,
                        mmap=True,
                    )
                )
            checkpoint = {}
            for key in cps[0].keys():
                if not torch.allclose(cps[0][key], cps[1][key]):
                    values = (cps[0][key], cps[1][key], cps[2][key], cps[3][key])
                    if "wo" in key or "w2" in key:
                        # Concat on dim=1 for "wo" and "w2".
                        checkpoint[key] = torch.cat(values, dim=1)
                    else:
                        # Concat on dim=0 for everything else.
                        checkpoint[key] = torch.cat(values, dim=0)
                else:
                    # Do not duplicate layers shared between each checkpoint.
                    checkpoint[key] = cps[0][key]
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device, mmap=True)
        fairseq2_checkpoint = kwargs.get("fairseq2", False)
        if fairseq2_checkpoint:
            print("Using fairseq2 checkpoint")
            checkpoint = convert_to_llama_checkpoint(checkpoint=checkpoint)
        if "model" in checkpoint:
            # NB: some checkpoint contains a "model" field, which is the actual weights dict
            checkpoint = checkpoint["model"]

        if (not fairseq2_checkpoint) and checkpoint.get(
            "final_proj.weight", None
        ) is not None:
            print(
                """

************************************************************
This looks like a Fairseq2 checkpoint (based on the presence
of `final_proj.weight`.

You can import Fairseq2 checkpoints using the --fairseq2
option, but --fairseq2 was not specified.  Please verify
the checkpoint format to avoid generating faulty models.
************************************************************
"""
            )

        # get checkpoint dtype
        self.dtype = None
        if len(checkpoint) > 0:
            first_key = next(iter(checkpoint))
            first = checkpoint[first_key]
            self.dtype = first.dtype
            mismatched_dtypes = [
                (key, value.dtype)
                for key, value in checkpoint.items()
                if value.dtype != self.dtype
            ]
            if len(mismatched_dtypes) > 0:
                print(
                    f"Mixed dtype model. Dtype of {first_key}: {first.dtype}. Mismatches in the checkpoint: {mismatched_dtypes}"
                )
        with open(params_path, "r") as f:
            params = json.loads(f.read())
        max_seq_len = self.max_seq_len
        max_batch_size = 1
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            use_kv_cache=self.use_kv_cache,
            use_sdpa_with_kv_cache_op=self.use_sdpa_with_kv_cache_op,
            generate_full_logits=self.generate_full_logits,
            enable_dynamic_shape=self.enable_dynamic_shape,
            **params,
        )
        if kwargs.get("fairseq2", False):
            print("Using fairseq2 checkpoint")
            checkpoint = convert_to_llama_checkpoint(checkpoint=checkpoint)
        if kwargs.get("verbose", False):
            print("============= weights ================")
            print("{key} : {weights.numel()} : {weights.size()}")
            for key, weights in checkpoint.items():
                print(f"{key} : {weights.numel()} : {weights.size()}")
            print("============= /weights ================")

        # Within the device="meta" context, tensors that are created do not carry data.
        # They possess all other metadata a tensor carries such as size, stride, requires_grad.
        with torch.device("meta"):
            self.model_ = Transformer(model_args)

        if "int8" in str(checkpoint_path):
            print("Using int8 weight-only quantization!")
            # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.examples.models.source_transformation.quantize`
            from ..source_transformation.quantize import WeightOnlyInt8QuantHandler

            simple_quantizer = WeightOnlyInt8QuantHandler(self.model_)
            self.model_ = simple_quantizer.convert_for_runtime()
        elif "8da4w" in str(checkpoint_path):
            print("Using int4 weight and int8 dynamic activation quantization!")
            from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer

            self.model_ = Int8DynActInt4WeightQuantizer()._convert_for_runtime(
                self.model_
            )
        elif hasattr(self.args, "use_spin_quant") and self.args.use_spin_quant:
            print("Using SPIN quantization.")
            assert hasattr(self.args, "group_size"), "group_size must be specified"
            assert hasattr(
                self.args, "quantization_mode"
            ), "quantization_mode must be specified"
            assert hasattr(
                self.args, "dtype_override"
            ), "dtype_override must be specified"
            from .source_transformation.spin_quant import (
                sanitize_checkpoint_from_spinquant,
                transform_for_spinquant,
            )

            mapping = {
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }

            self.model_ = transform_for_spinquant(
                self.model_,
                checkpoint,
                self.args.group_size,
                self.args.quantization_mode,
                mapping[self.args.dtype_override],
            )

            sanitize_checkpoint_from_spinquant(
                checkpoint,
                self.args.group_size,
            )

        # assign=True: load params/buffers by assignment instead of performing an in-place copy.
        # Because we are using device="meta", tensors do not have memory associated with them
        # and an in-place copy is a no-op. Use assign=True in load_state_dict for this scenario.
        missing, unexpected = self.model_.load_state_dict(
            checkpoint,
            strict=False,
            assign=True,
        )  # self.model_ = Transformer(gptconf)
        if kwargs.get("verbose", False):
            print("============= missing keys ================")
            print(missing)
            print("============= /missing ================")
            print("============= unexpected keys ================")
            print(unexpected)
            print("============= /unexpected ================")

    def get_eager_model(self):
        if self.dtype:
            # convert to the type of the provided checkpoint
            # input and output are torch.long, so signature unchanged
            return self.model_.to(self.dtype)
        else:
            # int8 quantization code has some bf16,
            # switch all to FP32
            return self.model_.to(torch.float32)

    def get_example_inputs(self):
        if self.use_kv_cache:
            return self.get_example_inputs_kvcache_sdpa()
        else:
            return (
                torch.tensor(
                    [[1, 2, 3]], dtype=torch.long
                ),  # tokens, with kv cache our input token length is always just 1 token.
            )

    # assumption is the custom op doesnt support dynamic shape right now. It might but its untested so lets first get static shape working
    def get_example_inputs_kvcache_sdpa(self):
        if self.enable_dynamic_shape:
            return (
                torch.tensor([[2, 3, 4]], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
            )
        else:
            return (
                torch.tensor(
                    [[1]], dtype=torch.long
                ),  # tokens, with kv cache our input token length is always just 1 token.
                torch.tensor(
                    [0], dtype=torch.long
                ),  # start_pos, what token of output are we on.
            )
class EagerModelFactory:
    """
    A factory class for dynamically creating instances of classes implementing EagerModelBase.
    """

    @staticmethod
    def create_model(
        module_name, model_class_name, **kwargs
    ) -> Tuple[torch.nn.Module, Any, Any]:
        """
        Create an instance of a model class that implements EagerModelBase and retrieve related data.

        Args:
            module_name (str): The name of the module containing the model class.
            model_class_name (str): The name of the model class to create an instance of.

        Returns:
            Tuple[nn.Module, Any]: A tuple containing the eager PyTorch model instance and example inputs,
              and any dynamic shape information for those inputs.

        Raises:
            ValueError: If the provided model class is not found in the module.
        """
        package_prefix = "executorch." if not os.getcwd().endswith("executorch") else ""
        module = importlib.import_module(
            f"{package_prefix}examples.models.{module_name}"
        )
        print(">>>", module, model_class_name)
        model = Llama2Model(**kwargs)
        if hasattr(model, "get_dynamic_shapes"):
            return (
                model.get_eager_model(),
                model.get_example_inputs(),
                model.get_dynamic_shapes(),
            )
        else:
            return model.get_eager_model(), model.get_example_inputs(), None

IS_FBCODE = True  #  os.environ.get("FBCODE_PLATFORM", False)
FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

pkg_name = __name__
verbosity_setting = None


class WeightType(Enum):
    LLAMA = "LLAMA"
    FAIRSEQ2 = "FAIRSEQ2"


def set_pkg_name(name: str) -> None:
    global pkg_name
    pkg_name = name


def get_resource_path(resource_name) -> str:
    return pkg_resources.resource_filename(pkg_name, resource_name)


def set_verbosity(val):
    global verbosity_setting
    verbosity_setting = val


def verbose_export():
    return verbosity_setting


def build_model(
    modelname: str = "model",
    extra_opts: str = "",
    *,
    par_local_output: bool = False,
    resource_pkg_name: str = __name__,
) -> str:
    if False:  # par_local_output:
        output_dir_path = "par:."
    else:
        output_dir_path = "."

    argString = f"--checkpoint par:{modelname}_ckpt.pt --params par:{modelname}_params.json {extra_opts} --output-dir {output_dir_path}"
    parser = build_args_parser()
    args = parser.parse_args(shlex.split(argString))
    # pkg_name = resource_pkg_name
    return export_llama(modelname, args)


def build_args_parser() -> argparse.ArgumentParser:
    ckpt_dir = f"{Path(__file__).absolute().parent.as_posix()}"
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", default=".", help="output directory")
    # parser.add_argument(
    #     "-q", "--quantized_ckpt", default=None, help="quantized checkpoint file"
    # )
    parser.add_argument(
        "-E",
        "--embedding-quantize",
        default=None,
        type=str,
        help="type of embedding quantization, '<bitwidth>,<groupsize>', e.g., '8,1024'.",
    )
    parser.add_argument(
        "--pt2e_quantize",
        default=None,
        choices=[
            "xnnpack_dynamic",
            "xnnpack_dynamic_qc4",
            "qnn_8a8w",
            "qnn_16a16w",
            "qnn_16a4w",
            "coreml_c4w",
            "coreml_8a_c8w",
            "coreml_8a_c4w",
            "coreml_baseline_8a_c8w",
            "coreml_baseline_8a_c4w",
        ],
        help="Use PT2E quantization. Comma separated options. e.g. xnnpack_dynamic (for per channel 8 bit weight), xnnpack_dynamic_qc4 (for per channel 4 bit weight), embedding.",
    )
    parser.add_argument(
        "-qmode",
        "--quantization_mode",
        type=str,
        default=None,
        choices=["int8", "8da4w", "8da4w-gptq"],
        help="type of quantization",
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        default=f"{ckpt_dir}/params/demo_rand_params.pth",
        help="checkpoint path",
    )

    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        help="checkpoint directory. Use with a sharded checkpoint, not for the standard llama2 model. Note, checkpoint_dir takes precedence over checkpoint if both are set.",
    )

    parser.add_argument(
        "--calibration_tasks",
        nargs="+",
        type=str,
        default=None,
        help="Tasks for GPTQ calibration from lm_eval",
    )
    parser.add_argument(
        "--calibration_limit",
        type=int,
        default=None,
        help="number of samples used for calibration from lm_eval",
    )
    parser.add_argument(
        "--calibration_seq_length",
        type=int,
        default=None,
        help="Sequence length for GPTQ calibration from lm_eval",
    )
    parser.add_argument(
        "--calibration_data",
        type=str,
        default="Once upon a time",
        help="Calibration prompts from users",
    )
    parser.add_argument(
        "-t",
        "--tokenizer_path",
        default=None,
        help="tokenizer path (Note: .model not .bin)",
    )
    parser.add_argument(
        "-kv",
        "--use_kv_cache",
        default=False,
        action="store_true",
        help="Whether or not to export a model using kv cache",
    )
    parser.add_argument(
        "--num_sharding",
        type=int,
        default=0,
        help="Specify the number of splits by inserting the fallback custom op. The graph will be split evenly by layers.",
    )
    parser.add_argument(
        "--use_sdpa_with_kv_cache",
        default=False,
        action="store_true",
        help="Whether to use sdpa_with_kv_cache update op when using kv cache",
    )
    parser.add_argument(
        "--disable_dynamic_shape",
        dest="enable_dynamic_shape",
        default=True,  # Enable this by default
        action="store_false",
        help="Enable dynamic shape along seq dim. Used for faster prefill",
    )
    parser.add_argument(
        "-p",
        "--params",
        default=f"{ckpt_dir}/params/demo_config.json",
        help="config.json",
    )
    parser.add_argument(
        "--optimized_rotation_path",
        default=None,
        required=False,
        help="[QNN backend] Optimized rotation checkpoint path. Just apply R1/R2 here."
        "You can download the optimized rotation matrices from https://github.com/facebookresearch/SpinQuant/tree/main",
    )
    parser.add_argument(
        "-m",
        "--metadata",
        default=None,
        help='metadata string in json format. Example {"key": 1, "key2": "value2"}',
    )
    parser.add_argument(
        "-s",
        "--so_library",
        default=None,
        required=False,
        help="shared library for quantized operators",
    )
    parser.add_argument(
        "--profile_memory",
        required=False,
        action="store_true",
        help="Generate chrome trace of activation memory for intermediate tensors.",
    )
    parser.add_argument(
        "-prof",
        "--profile_path",
        default=None,
        help="Use cProfile to profile model export. Results saved to profile_path as a html file.",
    )
    parser.add_argument(
        "-G",
        "--group_size",
        type=int,
        default=None,
        help="group_size for weight quantization",
    )

    parser.add_argument(
        "-d",
        "--dtype-override",
        default="fp32",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        help="Override the dtype of the model (default is the checkpoint dtype)."
        "Options: fp32, fp16, bf16. Please be aware that only some backends support fp16 and bf16.",
    )

    parser.add_argument(
        "-n",
        "--output_name",
        default=None,
        help="Override the output filename of the saved pte model file.",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="maximum length sequence to evaluate",
    )

    parser.add_argument("-2", "--fairseq2", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-X", "--xnnpack", action="store_true")
    parser.add_argument("-V", "--vulkan", action="store_true")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--coreml", action="store_true")
    parser.add_argument(
        "--coreml-enable-state",
        action="store_true",
        help="This option is only for coreml, and is only supported for MacOS15+/iOS18+",
    )
    parser.add_argument(
        "--coreml-preserve-sdpa",
        action="store_true",
        help="This option is only for coreml: Preserve sdpa in torch edge program to use coreml iOS18.sdpa op",
    )
    parser.add_argument(
        "--coreml-quantize",
        default=None,
        choices=["b4w"],
        help="This option is only for coreml: Use coreml quantization, e.g. b4w (for blockwise 4 bit weight)",
    )
    parser.add_argument(
        "--coreml-ios",
        type=int,
        default=15,
        choices=(15, 16, 17, 18),
        help="This option is only for coreml: The minimum iOS version to deploy",
    )
    parser.add_argument(
        "--qnn",
        action="store_true",
        help="Delegate llama2 to qnn backend (Qualcomm), please use it --kv_cahce=True",
    )

    parser.add_argument(
        "--expand_rope_table",
        default=False,
        action="store_true",
        help="[Temp workaround] Expand sin/cos table in head dim to take vectorized path in optimized kernels.",
    )

    parser.add_argument(
        "--generate_etrecord",
        action="store_true",
        required=False,
        default=False,
        help="Generate the ETRecord debug artifact.",
    )

    parser.add_argument(
        "--generate_full_logits",
        action="store_true",
        required=False,
        default=False,
        help="Generate logits for all inputs.",
    )

    parser.add_argument(
        "--soc_model",
        help="[QNN backend] SoC model of current device. e.g. 'SM8650' for Snapdragon 8 Gen 3.",
        type=str,
        required=False,
        default="SM8650",
    )

    parser.add_argument(
        "-sq",
        "--use_spin_quant",
        type=str,
        default=None,
        choices=["cuda", "native"],
        help="Use SpinQuant for better quantization performance. Only support cuda and native.",
    )
    return parser


def canonical_path(path: Union[str, Path], *, dir: bool = False) -> str:
    path = str(path)

    if verbose_export():
        print(f"creating canonical path for {path}")

    if not path.startswith("par:"):
        return path

    if not IS_FBCODE:
        print("not FBCODE")
        return path[4:]
    else:
        return_val = pkg_resources.resource_filename(pkg_name, path[4:])
        if verbose_export():
            print(f"canonical name is: {return_val}")
        return return_val


def export_llama(modelname, args) -> str:
    if args.profile_path is not None:
        try:
            from executorch.util.python_profiler import CProfilerFlameGraph

            with CProfilerFlameGraph(args.profile_path):
                builder = _export_llama(modelname, args)
                assert (
                    filename := builder.get_saved_pte_filename()
                ) is not None, "Fail to get file name from builder"
                return filename
        except ImportError:
            print(
                "Please run `pip install snakeviz` to install required dependencies for cProfiler flamegraph."
            )
            return ""
    else:
        builder = _export_llama(modelname, args)
        assert (
            filename := builder.get_saved_pte_filename()
        ) is not None, "Fail to get file name from builder"
        return filename


def _prepare_for_llama_export(modelname: str, args) -> LLMEdgeManager:
    """
    Helper function for export_llama. Loads the model from checkpoint and params,
    and sets up a LLMEdgeManager with initial transforms and dtype conversion.

    Returns a LLMEdgeManager prior to calling export_to_edge with quantizers
    """

    print("HERE")
    # load model from checkpoint and params.json
    checkpoint_path = canonical_path(args.checkpoint) if args.checkpoint else None
    checkpoint_dir = (
        canonical_path(args.checkpoint_dir) if args.checkpoint_dir else None
    )
    params_path = canonical_path(args.params)
    output_dir_path = canonical_path(args.output_dir, dir=True)
    weight_type = WeightType.FAIRSEQ2 if args.fairseq2 else WeightType.LLAMA

    # dtype override
    if args.dtype_override is not None:
        dtype_override = DType[args.dtype_override]
    elif args.quantization_mode in ["8da4w", "8da4w-gptq"]:
        dtype_override = DType["fp16"]
    else:
        dtype_override = None

    model = (
        _load_llama_model(
            modelname=modelname,
            checkpoint=checkpoint_path,
            checkpoint_dir=checkpoint_dir,
            params_path=params_path,
            use_kv_cache=args.use_kv_cache,
            use_sdpa_with_kv_cache=args.use_sdpa_with_kv_cache,
            generate_full_logits=args.generate_full_logits,
            weight_type=weight_type,
            enable_dynamic_shape=args.enable_dynamic_shape,
            calibration_tasks=args.calibration_tasks,
            calibration_limit=args.calibration_limit,
            calibration_seq_length=args.calibration_seq_length,
            calibration_data=args.calibration_data,
            tokenizer_path=args.tokenizer_path,
            verbose=args.verbose,
            max_seq_len=args.max_seq_length,
            metadata_str=args.metadata,
            args=args,
        )
        .set_output_dir(output_dir_path)
        .to_dtype(dtype_override)
        # .source_transform(_get_source_transforms(modelname, dtype_override, args))
    )
    print(model)
    return model


def get_quantizer_and_quant_params(args):
    pt2e_quant_params = get_pt2e_quantization_params(
        args.pt2e_quantize, args.quantization_mode
    )
    quantizers = get_pt2e_quantizers(pt2e_quant_params, args.so_library)
    quant_dtype = None
    if args.qnn and args.pt2e_quantize:
        assert len(quantizers) == 0, "Should not enable both xnnpack and qnn"
        qnn_quantizer, quant_dtype = get_qnn_quantizer(
            args.pt2e_quantize, args.quantization_mode
        )
        quantizers.append(qnn_quantizer)
    if args.coreml and args.pt2e_quantize:
        assert len(quantizers) == 0, "Should not enable both xnnpack / qnn and coreml"
        coreml_quantizer = get_coreml_quantizer(args.pt2e_quantize)
        quantizers.append(coreml_quantizer)
    logging.info(f"Applying quantizers: {quantizers}")
    return pt2e_quant_params, quantizers, quant_dtype


def _validate_args(args):
    """
    TODO: Combine all the backends under --backend args
    """
    if args.enable_dynamic_shape and (args.coreml or args.mps or args.qnn):
        raise ValueError(
            "Dynamic shape is not supported with coreml, MPS or qnn backends."
            " Please use --disable_dynamic_shape."
        )

    if args.num_sharding > 0 and not args.qnn:
        raise ValueError("Model shard is only supported with qnn backend now.")


def _export_llama(modelname, args) -> LLMEdgeManager:  # noqa: C901
    _validate_args(args)
    pt2e_quant_params, quantizers, quant_dtype = get_quantizer_and_quant_params(args)

    # export_to_edge
    builder_exported_to_edge = (
        _prepare_for_llama_export(modelname, args)
        .capture_pre_autograd_graph()
        #.pt2e_quantize(quantizers)
        .export_to_edge()
    )

    modelname = builder_exported_to_edge.modelname

    # to_backend
    partitioners = []
    if pt2e_quant_params is not None and pt2e_quant_params.quantize_linear is not None:
        partitioners.append(get_xnnpack_partitioner())
        modelname = f"xnnpack_dq_{modelname}"

    if args.xnnpack:
        partitioners.append(get_xnnpack_partitioner())
        modelname = f"xnnpack_{modelname}"

    if args.vulkan:
        partitioners.append(
            get_vulkan_partitioner(
                args.dtype_override,
                args.quantization_mode,
            )
        )
        modelname = f"vulkan_{modelname}"

    if args.mps:
        partitioners.append(get_mps_partitioner(args.use_kv_cache))
        modelname = f"mps_{modelname}"

    if args.coreml:
        coreml_partitioner = get_coreml_partitioner(
            args.coreml_ios,
            args.embedding_quantize,
            args.pt2e_quantize,
            args.coreml_quantize,
        )
        partitioners.append(coreml_partitioner)
        modelname = f"coreml_{modelname}"

    if args.qnn:
        from executorch.extension.llm.custom_ops import model_sharding

        partitioners.append(
            get_qnn_partitioner(
                args.use_kv_cache, args.pt2e_quantize, args.num_sharding, args.soc_model
            )
        )
        # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.qualcomm.utils.utils`
        from executorch.backends.qualcomm.utils.utils import _transform

        # pyre-ignore: Undefined attribute [16]: Module `executorch.backends` has no attribute `qualcomm`, Optional type has no attribute `exported_program`
        _transform(builder_exported_to_edge.edge_manager.exported_program())

        if args.num_sharding > 0:
            model_sharding.split_graph(
                builder_exported_to_edge.edge_manager.exported_program(),
                builder_exported_to_edge.metadata["get_n_layers"],
                shares=args.num_sharding,
            )

    if args.generate_etrecord:
        if not builder_exported_to_edge.edge_manager:
            raise ValueError("Unable to generate etrecord due to missing edge manager.")

        logging.info("Generating etrecord")
        # Copy the edge manager which will be serialized into etrecord. This is memory-wise expensive.
        edge_manager_copy = copy.deepcopy(builder_exported_to_edge.edge_manager)
        print("PART2", partitioners)
        builder = builder_exported_to_edge.to_backend(partitioners)
        if args.num_sharding > 0 and args.qnn:
            from executorch.backends.qualcomm.utils.utils import canonicalize_program

            canonicalize_program(builder.edge_manager.exported_program())

        builder = builder.to_executorch()

        # Generate ETRecord
        if edge_manager_copy:
            generate_etrecord(
                et_record="etrecord.bin",
                edge_dialect_program=edge_manager_copy,
                executorch_program=builder.export_program,
            )
            logging.info("Generated etrecord.bin")
    else:
        print("PART", partitioners)
        builder = builder_exported_to_edge.to_backend(partitioners)
        if args.num_sharding > 0 and args.qnn:
            from executorch.backends.qualcomm.utils.utils import canonicalize_program

            canonicalize_program(builder.edge_manager.exported_program())

        builder = builder.to_executorch()

    if args.profile_memory:
        generate_memory_trace(builder.export_program, "memory_profile.json")

    if builder.dtype == DType.fp16:
        modelname = f"{modelname}_h"

    if args.output_name:
        modelname = args.output_name
        if modelname.endswith(".pte"):
            output_file = modelname
            modelname = modelname[:-4]
            print(f"modelname: {modelname}")
            print(f"output_file: {output_file}")
        else:
            output_file = f"{builder.output_dir}/{modelname}.pte"
            print(f"modelname: {modelname}")
            print(f"output_file: {output_file}")
    else:
        output_file = f"{builder.output_dir}/{modelname}.pte"

    builder.save_to_pte(output_file)

    return builder


def _load_llama_model_metadata(
    weight_type: WeightType,
    use_kv_cache: bool,
    use_sdpa_with_kv_cache: bool,
    enable_dynamic_shape: bool,
    model_args: ModelArgs,
    metadata_str: Optional[str] = None,
):
    is_fairseq2 = weight_type == WeightType.FAIRSEQ2
    metadata = {
        "append_eos_to_prompt": is_fairseq2,  # For language llama, tell the runtime to always append EOS token(s) to prompt.
        "get_bos_id": 3 if is_fairseq2 else 1,
        "get_eos_ids": [3] if is_fairseq2 else [2],
        "get_max_seq_len": model_args.max_seq_len,
        "get_n_bos": 1,
        "get_n_eos": 2 if is_fairseq2 else 1,
        "get_n_layers": model_args.n_layers,
        "get_vocab_size": model_args.vocab_size,
        "use_kv_cache": use_kv_cache,
        "use_sdpa_with_kv_cache": use_sdpa_with_kv_cache,
        "enable_dynamic_shape": enable_dynamic_shape,
    }
    if metadata_str:
        try:
            extra = json.loads(metadata_str)
            for k, v in extra.items():
                metadata[k] = v
        except JSONDecodeError:
            logging.error("Invalid metadata, should be a valid JSON string")
    return metadata


def _load_llama_model(
    *,
    modelname: str = "llama2",
    checkpoint: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    params_path: str,
    use_kv_cache: bool = False,
    use_sdpa_with_kv_cache: bool = False,
    generate_full_logits: bool = False,
    weight_type: WeightType = WeightType.LLAMA,
    enable_dynamic_shape: bool = False,
    calibration_tasks: Optional[List[str]] = None,
    calibration_limit: Optional[int] = None,
    calibration_seq_length: Optional[int] = None,
    calibration_data: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    verbose: bool = False,
    max_seq_len: int = 128,
    metadata_str: Optional[str] = None,
    args,
) -> "LLMEdgeManager":
    """
    A helper util that builds a Llama2 model. It returns a LLMEdgeManager that
    can help further lower the model to ExecuTorch.
    Returns:
        An instance of LLMEdgeManager which contains the eager mode model.
    """
    assert (
        checkpoint or checkpoint_dir
    ) and params_path, "Both checkpoint/checkpoint_dir and params can't be empty"
    logging.info(
        f"Loading model with checkpoint={checkpoint}, params={params_path}, use_kv_cache={use_kv_cache}, weight_type={weight_type}"
    )
    model, example_inputs, _ = EagerModelFactory.create_model(
        "llama2",
        "Llama2Model",
        checkpoint=checkpoint,
        checkpoint_dir=checkpoint_dir,
        params=params_path,
        use_kv_cache=use_kv_cache,
        use_sdpa_with_kv_cache=use_sdpa_with_kv_cache,
        generate_full_logits=generate_full_logits,
        fairseq2=weight_type == WeightType.FAIRSEQ2,
        max_seq_len=max_seq_len,
        enable_dynamic_shape=enable_dynamic_shape,
        args=args,
    )
    state_dict = model.state_dict()
    dtype = state_dict[next(iter(state_dict))].dtype
    assert dtype in [
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ], f"Only support bfloat16, fp16 or fp32 got {dtype}"
    logging.info(f"Loaded model with dtype={dtype}")
    print(model)

    if dtype == torch.bfloat16:
        dtype = DType.bf16
    elif dtype == torch.float16:
        dtype = DType.fp16
    elif dtype == torch.float32:
        dtype = DType.fp32
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    return LLMEdgeManager(
        model=model,
        modelname=modelname,
        max_seq_len=model.params.max_seq_len,
        dtype=dtype,
        use_kv_cache=use_kv_cache,
        generate_full_logits=generate_full_logits,
        example_inputs=example_inputs,
        enable_dynamic_shape=enable_dynamic_shape,
        calibration_tasks=calibration_tasks,
        calibration_limit=calibration_limit,
        calibration_seq_length=calibration_seq_length,
        calibration_data=calibration_data,
        tokenizer_path=tokenizer_path,
        verbose=verbose,
        metadata=_load_llama_model_metadata(
            weight_type,
            use_kv_cache,
            use_sdpa_with_kv_cache,
            enable_dynamic_shape,
            model.params,
            metadata_str,
        ),
        args=args,
    )

import logging

import torch

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def main() -> None:
    seed = 42
    torch.manual_seed(seed)
    modelname = "llama2"
    parser = build_args_parser()
    args = parser.parse_args()
    export_llama(modelname, args)


if __name__ == "__main__":
    main()  # pragma: no cover
