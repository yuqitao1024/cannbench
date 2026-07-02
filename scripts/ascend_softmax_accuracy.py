#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
except ImportError as exc:
    raise SystemExit("torch_npu is required on the Ascend test machine") from exc


DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
}

DATASET_NAMES = ("smoke", "realistic", "stress")

SOFTMAX_CASES = (
    ("smoke", "tiny_logits", (32, 128), -1, "smoke_fixture"),
    ("smoke", "tiny_attention_scores", (2, 4, 8, 8), -1, "smoke_attention_fixture"),
    ("smoke", "tiny_channel_softmax", (2, 16, 8, 8), 1, "smoke_channel_fixture"),
    ("realistic", "t5_attention", (4, 8, 1024, 1024), -1, "T5Small"),
    ("realistic", "xcit_attention", (4, 16, 48, 48), -1, "xcit_large_24_p8_224"),
    ("realistic", "convbert_attention", (16, 6, 512, 512), -1, "YituTechConvBert"),
    ("realistic", "convbert_local_kernel", (49152, 9, 1), 1, "YituTechConvBert"),
    ("realistic", "deberta_attention", (2, 24, 512, 512), -1, "DebertaV2ForMaskedLM"),
    ("realistic", "electra_attention", (32, 4, 512, 512), -1, "ElectraForCausalLM"),
    ("realistic", "gptneo_attention", (32, 16, 128, 128), -1, "GPTNeoForCausalLM"),
    ("realistic", "gptj_attention", (1, 16, 128, 128), -1, "GPTJForCausalLM"),
    ("realistic", "layoutlm_attention", (16, 12, 512, 512), -1, "LayoutLMForMaskedLM"),
    ("realistic", "mobilebert_attention", (128, 4, 128, 128), -1, "MobileBertForMaskedLM"),
    ("realistic", "xlnet_attention", (8, 16, 512, 512), 3, "XLNetLMHeadModel"),
    ("realistic", "plbart_attention", (96, 1024, 1024), -1, "PLBartForCausalLM"),
    ("realistic", "pegasus_attention", (512, 128, 128), -1, "PegasusForConditionalGeneration"),
    ("realistic", "trocr_attention", (512, 256, 256), -1, "TrOCRForCausalLM"),
    ("realistic", "levit_global_attention", (1024, 4, 196, 196), -1, "levit_128"),
    ("realistic", "levit_mixed_attention", (1024, 8, 49, 196), -1, "levit_128"),
    ("realistic", "swin_window_attention", (256, 16, 49, 49), -1, "swin_base_patch4_window7_224"),
    ("realistic", "crossvit_cls_attention", (256, 4, 1, 197), -1, "crossvit_9_240"),
    ("realistic", "halonet_window_attention", (1024, 4, 64, 144), -1, "eca_halonext26ts"),
    ("realistic", "speech_transformer_attention", (80, 204, 204), 2, "speech_transformer"),
    ("realistic", "bert_pytorch_attention", (16, 12, 128, 128), -1, "BERT_pytorch"),
    ("realistic", "t5_logits", (4096, 32128), 1, "T5Small"),
    ("realistic", "convbert_logits", (8192, 30522), 1, "YituTechConvBert"),
    ("realistic", "longformer_logits", (4096, 50265), 1, "AllenaiLongformerBase"),
    ("realistic", "plbart_logits", (8192, 50005), 1, "PLBartForCausalLM"),
    ("realistic", "camembert_logits", (8192, 32005), 1, "CamemBert"),
    ("realistic", "m2m100_logits", (2048, 128112), 1, "M2M100ForConditionalGeneration"),
    ("realistic", "mt5_logits", (2048, 250112), 1, "MT5ForConditionalGeneration"),
    ("realistic", "xglm_logits", (1024, 256008), 1, "XGLMForCausalLM"),
    ("realistic", "opt_logits", (4094, 50272), 1, "OPTForCausalLM"),
    ("stress", "long_context_attention", (1, 32, 4096, 4096), -1, "llm_attention_boundary"),
    ("stress", "wide_vocab_lm_logits", (8192, 128256), 1, "llm_logits_boundary"),
    ("stress", "moe_router_scores", (4096, 128), -1, "moe_router_boundary"),
    ("stress", "small_reduction_axis", (16384, 2), -1, "softmax_small_axis_boundary"),
    ("stress", "vision_window_batch", (2048, 16, 49, 49), -1, "vision_window_batch_boundary"),
    ("stress", "channelwise_activation_map", (64, 2048, 7, 7), 1, "channel_activation_boundary"),
    ("stress", "beam_search_token_scores", (512, 4, 64000), -1, "beam_search_token_boundary"),
)


@dataclass
class RunResult:
    name: str
    output: torch.Tensor


def synchronize() -> None:
    torch.npu.synchronize()


def load_simt_op(package_name: str) -> Callable[[torch.Tensor, int], torch.Tensor]:
    try:
        simt_ops = __import__(f"{package_name}.ops", fromlist=["ops"])
    except ImportError as exc:
        raise SystemExit(
            f"{package_name} is not installed. Install the target SIMT package first."
        ) from exc

    if not simt_ops.is_extension_loaded() or not simt_ops.has_spatial_softmax_forward_op():
        raise SystemExit(
            f"{package_name} extension is not loaded or spatial_softmax_forward is not registered"
        )

    def run(input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
        return simt_ops.spatial_softmax_forward(input_tensor, dim, False)

    return run


def cann_ops_softmax(input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
    return F.softmax(input_tensor, dim=dim)


def run_op(
    name: str,
    fn: Callable[[torch.Tensor, int], torch.Tensor],
    input_tensor: torch.Tensor,
    dim: int,
    warmup: int,
    iters: int,
) -> RunResult:
    output = None
    with torch.no_grad():
        for _ in range(warmup):
            output = fn(input_tensor, dim)
        synchronize()

        for _ in range(iters):
            output = fn(input_tensor, dim)
        synchronize()

    assert output is not None
    return RunResult(name=name, output=output)


def output_stats(name: str, output: torch.Tensor, dim: int) -> None:
    data = output.detach().to("cpu", dtype=torch.float32)
    finite = torch.isfinite(data)
    nan_count = torch.isnan(data).sum().item()
    inf_count = torch.isinf(data).sum().item()
    finite_count = finite.sum().item()
    total = data.numel()
    print(f"{name} output:")
    print(f"  elements: {total}")
    print(f"  finite: {finite_count}")
    print(f"  nan: {nan_count}")
    print(f"  inf: {inf_count}")
    if finite_count:
        finite_values = data[finite]
        print(f"  finite_min: {finite_values.min().item():.8g}")
        print(f"  finite_max: {finite_values.max().item():.8g}")
        row_sum = data.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).sum(dim=dim)
        row_sum_err = (row_sum - 1.0).abs()
        print(f"  max_abs_row_sum_error: {row_sum_err.max().item():.8g}")


def compare_outputs(reference: torch.Tensor, candidate: torch.Tensor, atol: float, rtol: float, dim: int) -> None:
    ref = reference.detach().to("cpu", dtype=torch.float32)
    got = candidate.detach().to("cpu", dtype=torch.float32)
    output_stats("reference_cann_ops_library", ref, dim)
    output_stats("candidate_simt", got, dim)

    finite = torch.isfinite(ref) & torch.isfinite(got)
    if finite.any():
        diff = (got[finite] - ref[finite]).abs()
        denom = ref[finite].abs().clamp_min(1e-6)
        max_abs = diff.max().item()
        max_rel = (diff / denom).max().item()
    else:
        max_abs = float("nan")
        max_rel = float("nan")
    allclose = torch.allclose(got, ref, atol=atol, rtol=rtol)
    print("accuracy:")
    print(f"  allclose: {allclose}  atol={atol} rtol={rtol}")
    print(f"  max_abs_error: {max_abs:.8g}")
    print(f"  max_rel_error: {max_rel:.8g}")


def accuracy_summary(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    atol: float,
    rtol: float,
) -> dict[str, float | int | bool]:
    ref = reference.detach().to("cpu", dtype=torch.float32)
    got = candidate.detach().to("cpu", dtype=torch.float32)
    finite = torch.isfinite(ref) & torch.isfinite(got)
    if finite.any():
        diff = (got[finite] - ref[finite]).abs()
        denom = ref[finite].abs().clamp_min(1e-6)
        max_abs = float(diff.max().item())
        max_rel = float((diff / denom).max().item())
    else:
        max_abs = float("nan")
        max_rel = float("nan")
    return {
        "allclose": bool(torch.allclose(got, ref, atol=atol, rtol=rtol)),
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "nan_count": int(torch.isnan(got).sum().item()),
        "inf_count": int(torch.isinf(got).sum().item()),
        "finite_count": int(torch.isfinite(got).sum().item()),
        "total": int(got.numel()),
    }


def parse_shape(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.replace("x", ",").split(",") if item.strip())


def load_cases() -> dict[str, dict[str, object]]:
    cases: dict[str, dict[str, object]] = {}
    for dataset, case_id, shape, dim, source_model in SOFTMAX_CASES:
        if case_id in cases:
            raise SystemExit(f"duplicate softmax case_id: {case_id}")
        cases[case_id] = {
            "dataset": dataset,
            "shape": shape,
            "dim": dim,
            "source_model": source_model,
        }
    return cases


def select_cases(
    all_cases: dict[str, dict[str, object]],
    dataset: str,
    case_name: str,
) -> dict[str, dict[str, object]]:
    selected = {
        name: case
        for name, case in all_cases.items()
        if dataset == "ALL" or case["dataset"] == dataset
    }
    if case_name == "ALL":
        return selected
    if case_name not in selected:
        raise SystemExit(f"case {case_name!r} is not available in dataset {dataset!r}")
    return {case_name: selected[case_name]}


def build_input(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device, seed: int) -> torch.Tensor:
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
    input_cpu = torch.randn(shape, generator=cpu_generator, dtype=dtype)
    input_tensor = input_cpu.to(device)
    synchronize()
    return input_tensor


def run_accuracy_case(
    case_name: str,
    case: dict[str, object],
    dtype_name: str,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    atol: float,
    rtol: float,
    simt_fn: Callable[[torch.Tensor, int], torch.Tensor],
    simt_label: str,
) -> dict[str, float | int | bool | str]:
    shape = tuple(case["shape"])
    dim = int(case["dim"])
    input_tensor = build_input(shape, dtype, device, seed)
    reference = run_op("cann_ops_library", cann_ops_softmax, input_tensor, dim, 0, 1).output
    candidate = run_op(simt_label, simt_fn, input_tensor, dim, 0, 1).output
    summary = accuracy_summary(reference, candidate, atol, rtol)
    return {
        **summary,
        "case": case_name,
        "dataset": str(case.get("dataset", "")),
        "shape": "x".join(str(item) for item in shape),
        "dim": dim,
        "dtype": dtype_name,
        "source_model": str(case.get("source_model", "")),
    }


def print_accuracy_table(rows: list[dict[str, object]]) -> None:
    print("accuracy_summary:")
    print("dataset,case,shape,dim,dtype,source_model,status,max_abs_error,max_rel_error,nan,inf,finite,total")
    for row in rows:
        print(format_accuracy_row(row))


def format_accuracy_row(row: dict[str, object]) -> str:
    status = "PASS" if row["allclose"] else "FAIL"
    return (
        f"{row['dataset']},{row['case']},{row['shape']},{row['dim']},{row['dtype']},"
        f"{row['source_model']},{status},{row['max_abs_error']:.8g},"
        f"{row['max_rel_error']:.8g},{row['nan_count']},{row['inf_count']},"
        f"{row['finite_count']},{row['total']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual Ascend softmax comparison for CANN ops library vs SIMT op.")
    parser.add_argument("--mode", choices=("cannops", "simt", "both"), default="both")
    parser.add_argument("--simt-package", default="aten_softmax_v2")
    parser.add_argument("--simt-label", default="simt_v2")
    parser.add_argument("--dataset", choices=["ALL", *DATASET_NAMES], default="ALL")
    parser.add_argument("--case", default="ALL", help="Case id to run, or ALL.")
    parser.add_argument("--shape", type=parse_shape, help="Override shape, e.g. 4,8,1024,1024 or 4x8x1024x1024")
    parser.add_argument("--dim", type=int, help="Override softmax dim")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float16")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-2)
    args = parser.parse_args()

    dtype = DTYPES[args.dtype]
    device = torch.device(args.device)

    torch.npu.set_device(device)
    cases = load_cases()
    selected_cases = select_cases(cases, args.dataset, args.case)

    if len(selected_cases) > 1:
        if args.mode != "both":
            raise SystemExit("batch accuracy requires --mode both")
        if args.shape is not None or args.dim is not None:
            raise SystemExit("--shape/--dim overrides are only supported for a single case")
        simt_fn = load_simt_op(args.simt_package)
        rows = []
        print("accuracy_summary:")
        print("dataset,case,shape,dim,dtype,source_model,status,max_abs_error,max_rel_error,nan,inf,finite,total", flush=True)
        for case_name, case in selected_cases.items():
            row = run_accuracy_case(
                case_name,
                case,
                args.dtype,
                dtype,
                device,
                args.seed,
                args.atol,
                args.rtol,
                simt_fn,
                args.simt_label,
            )
            rows.append(row)
            print(format_accuracy_row(row), flush=True)
        failed = sum(1 for row in rows if not row["allclose"])
        print(f"summary: total={len(rows)} passed={len(rows) - failed} failed={failed}")
        return 1 if failed else 0

    case_name, case = next(iter(selected_cases.items()))
    shape = args.shape or case["shape"]
    dim = args.dim if args.dim is not None else case["dim"]
    input_tensor = build_input(shape, dtype, device, args.seed)

    print("case:")
    print(f"  dataset: {case['dataset']}")
    print(f"  name: {case_name}")
    print(f"  shape: {shape}")
    print(f"  dim: {dim}")
    print(f"  dtype: {args.dtype}")
    print(f"  device: {args.device}")
    print("  input_source: cpu_random_then_to_npu")
    print(f"  warmup: {args.warmup}")
    print(f"  iters: {args.iters}")

    results: list[RunResult] = []
    if args.mode in ("cannops", "both"):
        results.append(run_op("cann_ops_library", cann_ops_softmax, input_tensor, dim, args.warmup, args.iters))
    if args.mode in ("simt", "both"):
        results.append(run_op(args.simt_label, load_simt_op(args.simt_package), input_tensor, dim, args.warmup, args.iters))

    if args.mode == "both" and len(results) == 2:
        compare_outputs(results[0].output, results[1].output, args.atol, args.rtol, dim)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
