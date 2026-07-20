import glob
import os
import shlex
import sysconfig
from distutils.errors import CompileError
from shutil import which

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

import torch
import torch.utils.cpp_extension as cpp_extension
import torch_npu


library_name = "aten_adaptive_max_pool3d_grad_v2"

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
        torch_npu_path,
        "include",
        "third_party",
        "acl",
        "inc",
    )
    torch_npu_lib = os.path.join(torch_npu_path, "lib")
    ascend_home = os.getenv("ASCEND_TOOLKIT_HOME") or os.getenv("ASCEND_HOME_PATH")
    ascend_asc_include = (
        os.path.join(ascend_home, "x86_64-linux", "asc", "include")
        if ascend_home
        else None
    )

    include_dirs = [
        *torch_include_paths,
        python_include,
        torch_npu_include,
        torch_npu_acl_include,
    ]
    if ascend_asc_include:
        include_dirs.append(ascend_asc_include)
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

        try:
            objects = []
            for source in ext.sources:
                source_ext = os.path.splitext(source)[1]
                object_path = os.path.join(
                    self.build_temp,
                    ext.name.replace(".", "_"),
                    os.path.relpath(source, EXTENSIONS_DIR) + ".o",
                )
                os.makedirs(os.path.dirname(object_path), exist_ok=True)
                if source_ext == ".asc":
                    objects.append(
                        self._compile_asc_source(
                            source,
                            object_path,
                            dep_paths["include_dirs"],
                            abi_value,
                            opt_flag,
                            debug_mode,
                        )
                    )
                elif source_ext in {".cpp", ".cc", ".cxx"}:
                    objects.append(
                        self._compile_cpp_source(
                            source,
                            object_path,
                            dep_paths["include_dirs"],
                            abi_value,
                            opt_flag,
                            debug_mode,
                        )
                    )
                else:
                    raise RuntimeError(f"Unsupported source type: {source}")
            self._link_extension(objects, ext_fullpath, dep_paths["library_dirs"])
        except Exception as exc:
            raise CompileError(str(exc)) from exc

    def _compile_asc_source(
        self,
        source,
        object_path,
        include_dirs,
        abi_value,
        opt_flag,
        debug_mode,
    ):
        compile_cmd = [
            "bisheng",
            "-x",
            "asc",
            "--enable-simt",
            f"--npu-arch={NPU_ARCH}",
            "-c",
            "-fPIC",
            "-std=c++17",
            opt_flag,
            f"-D_GLIBCXX_USE_CXX11_ABI={abi_value}",
            source,
            "-o",
            object_path,
        ]
        if debug_mode:
            compile_cmd.append("-g")
        for include_dir in include_dirs:
            compile_cmd.append(f"-I{include_dir}")
        self.spawn(compile_cmd)
        return object_path

    def _compile_cpp_source(
        self,
        source,
        object_path,
        include_dirs,
        abi_value,
        opt_flag,
        debug_mode,
    ):
        cxx = shlex.split(os.getenv("CXX") or sysconfig.get_config_var("CXX") or "c++")[0]
        compile_cmd = [
            cxx,
            "-c",
            "-fPIC",
            "-std=c++17",
            opt_flag,
            f"-D_GLIBCXX_USE_CXX11_ABI={abi_value}",
            source,
            "-o",
            object_path,
        ]
        if debug_mode:
            compile_cmd.append("-g")
        for include_dir in include_dirs:
            compile_cmd.append(f"-I{include_dir}")
        self.spawn(compile_cmd)
        return object_path

    def _link_extension(self, objects, ext_fullpath, library_dirs):
        cxx = shlex.split(os.getenv("CXX") or sysconfig.get_config_var("CXX") or "c++")[0]
        link_cmd = [cxx, "-shared", *objects]
        for library_dir in library_dirs:
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


def get_extensions():
    relative_extensions_dir = os.path.join(library_name, "csrc")
    sources = list(glob.glob(os.path.join(relative_extensions_dir, "*.cpp")))
    sources += list(glob.glob(os.path.join(relative_extensions_dir, "*.asc")))
    sources += list(glob.glob(os.path.join(relative_extensions_dir, "simt", "*.asc")))
    return [
        Extension(
            name=f"{library_name}._C",
            sources=sources,
            language="c++",
        )
    ]


setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch", "torch_npu"],
    description="CannBench AdaptiveMaxPool3DGrad SIMT v2 CANN arch35 migration",
    cmdclass={"build_ext": AscendBuildExtension},
)

