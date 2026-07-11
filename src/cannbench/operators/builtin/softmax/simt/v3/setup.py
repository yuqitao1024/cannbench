import glob
import os
import sysconfig
from distutils.errors import CompileError
from pathlib import Path
from shutil import which

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

import torch
import torch.utils.cpp_extension as cpp_extension
import torch_npu


library_name = "aten_softmax_v3"

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
EXTENSIONS_DIR = os.path.join(BASE_DIR, library_name, "csrc")
NPU_ARCH = os.getenv("NPU_ARCH")


def get_dependency_paths():
    python_include = sysconfig.get_config_var("INCLUDEPY")
    python_lib = sysconfig.get_config_var("LIBDIR")
    torch_include_paths = cpp_extension.include_paths()
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    ascend_home = os.getenv("ASCEND_HOME_PATH")
    ascend_include_candidates = []
    if ascend_home:
        ascend_include_candidates.extend(
            [
                os.path.join(ascend_home, "x86_64-linux", "asc", "include"),
                os.path.join(ascend_home, "include"),
            ]
        )
    ascend_include_candidates.extend(
        glob.glob("/usr/local/Ascend/cann-*/x86_64-linux/asc/include")
    )
    ascend_include_dirs = [
        path for path in ascend_include_candidates if os.path.isdir(path)
    ]

    torch_npu_path = os.path.dirname(torch_npu.__file__)
    torch_npu_include = os.path.join(torch_npu_path, "include")
    torch_npu_acl_include = os.path.join(
        torch_npu_path, "include", "third_party", "acl", "inc"
    )
    torch_npu_lib = os.path.join(torch_npu_path, "lib")

    include_dirs = [
        *torch_include_paths,
        python_include,
        *ascend_include_dirs,
        torch_npu_include,
        torch_npu_acl_include,
    ]
    library_dirs = [python_lib, torch_lib, torch_npu_lib]
    return {"include_dirs": include_dirs, "library_dirs": library_dirs}


class AscendBuildExtension(build_ext):
    def _check_bisheng_compiler(self):
        if not which("bisheng"):
            raise RuntimeError("bisheng command not found")

    def _common_compile_args(self, dep_paths, abi_value, opt_flag, debug_mode):
        args = [
            "bisheng",
            "-x",
            "asc",
            f"--npu-arch={NPU_ARCH}",
            "-fPIC",
            "-std=c++17",
            opt_flag,
            f"-D_GLIBCXX_USE_CXX11_ABI={abi_value}",
        ]
        if debug_mode:
            args.append("-g")
        for include_dir in dep_paths["include_dirs"]:
            args.append(f"-I{include_dir}")
        return args

    def _compile_sources_to_objects(
        self,
        sources,
        dep_paths,
        abi_value,
        opt_flag,
        debug_mode,
    ):
        os.makedirs(self.build_temp, exist_ok=True)
        object_files = []
        common_args = self._common_compile_args(
            dep_paths, abi_value, opt_flag, debug_mode
        )
        for source in sources:
            relative_source = os.path.relpath(source, BASE_DIR)
            object_name = relative_source.replace(os.sep, "_") + ".o"
            object_path = Path(self.build_temp) / object_name
            compile_cmd = [
                *common_args,
                "-c",
                source,
                "-o",
                str(object_path),
            ]
            self.spawn(compile_cmd)
            object_files.append(str(object_path))
        return object_files

    def _link_objects(self, ext_fullpath, object_files, dep_paths):
        link_cmd = [
            "bisheng",
            "-shared",
            "-fPIC",
            *object_files,
        ]
        for library_dir in dep_paths["library_dirs"]:
            link_cmd.append(f"-L{library_dir}")
        link_cmd.extend(
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
        self.spawn(link_cmd)

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

        try:
            object_files = self._compile_sources_to_objects(
                ext.sources,
                dep_paths,
                abi_value,
                opt_flag,
                debug_mode,
            )
            self._link_objects(ext_fullpath, object_files, dep_paths)
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
