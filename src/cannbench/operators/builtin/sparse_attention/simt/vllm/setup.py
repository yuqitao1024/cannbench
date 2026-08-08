from setuptools import find_packages, setup


setup(
    name="aten_dsa_sparse_attention_vllm",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["torch", "torch_npu"],
    description="CannBench-local wrapper for the copied vLLM-Ascend MLA kernel",
)
