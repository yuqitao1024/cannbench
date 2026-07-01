import glob
import os
import sysconfig
from distutils.errors import CompileError
from shutil import which

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

import torch
import torch.utils.cpp_extension as cpp_extension
import torch_npu


library_name = "aten_softmax_v2"

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
EXTENSIONS_DIR = os.path.join(BASE_DIR, library_name, "csrc")
NPU_ARCH = os.getenv("NPU_ARCH")


def get_dependency_paths():
    python_include = sysconfig.get_config_var("INCLUDEPY")
    python_lib = sysconfig.get_config_var("LIBDIR")
    torch_include_paths = cpp_extension.include_paths()
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")

    torch_npu_path = os.path.dirname(torch_npu.__file__)
    torch_npu_include = os.path.join(torch_npu_path, "include")
    torch_npu_acl_include = os.path.join(
        torch_npu_path, "include", "third_party", "acl", "inc"
    )
    torch_npu_lib = os.path.join(torch_npu_path, "lib")

    include_dirs = [
        *torch_include_paths,
        python_include,
        torch_npu_include,
        torch_npu_acl_include,
    ]
    library_dirs = [python_lib, torch_lib, torch_npu_lib]
    return {"include_dirs": include_dirs, "library_dirs": library_dirs}


class AscendBuildExtension(build_ext):
    def _check_bisheng_compiler(self):
        if not which("bisheng"):
            raise RuntimeError("bisheng command not found")

    def build_extension(self, ext):
        self._check_bisheng_compiler()
        if not NPU_ARCH:
            raise RuntimeError("NPU_ARCH environment variable is required")

        dep_paths = get_dependency_paths()
        ext_fullpath = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_fullpath), exist_ok=True)

        use_cxx11_abi = torch._C._GLIBCXX_USE_CXX11_ABI
        abi_value = "1" if use_cxx11_abi else "0"
        debug_mode = os.getenv("DEBUG", "0") == "1"
        opt_flag = "-O0" if debug_mode else "-O3"

        compile_cmd = [
            "bisheng",
            "-x",
            "asc",
            "--enable-simt",
            f"--npu-arch={NPU_ARCH}",
            "-shared",
            "-fPIC",
            "-std=c++17",
            opt_flag,
            f"-D_GLIBCXX_USE_CXX11_ABI={abi_value}",
            *ext.sources,
        ]

        if debug_mode:
            compile_cmd.append("-g")

        for include_dir in dep_paths["include_dirs"]:
            compile_cmd.append(f"-I{include_dir}")

        for library_dir in dep_paths["library_dirs"]:
            compile_cmd.append(f"-L{library_dir}")

        compile_cmd.extend(
            [
                "-ltorch_npu",
                "-ltorch_python",
                "-ltorch_cpu",
                "-ltorch",
                "-lc10",
                "-lm",
                "-o",
                ext_fullpath,
            ]
        )

        try:
            self.spawn(compile_cmd)
        except Exception as exc:
            raise CompileError(str(exc)) from exc


def get_extensions():
    sources = list(glob.glob(os.path.join(EXTENSIONS_DIR, "*.asc")))
    sources += list(glob.glob(os.path.join(EXTENSIONS_DIR, "simt", "*.asc")))
    return [
        Extension(
            name=f"{library_name}._C",
            sources=sources,
            language="asc",
        )
    ]


setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch", "torch_npu"],
    description="Ascend SIMT softmax migration helpers on torch_npu",
    cmdclass={"build_ext": AscendBuildExtension},
)
