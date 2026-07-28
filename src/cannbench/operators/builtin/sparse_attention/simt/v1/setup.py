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


library_name = "aten_dsa_sparse_attention"

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
EXTENSIONS_DIR = os.path.join(BASE_DIR, library_name, "csrc")
NPU_ARCH = os.getenv("NPU_ARCH")

HOST_SOURCES = [
    os.path.join(EXTENSIONS_DIR, "register.asc"),
    os.path.join(EXTENSIONS_DIR, "sparse_attention.asc"),
]
# Keep one VF entry per device ELF so msopprof can replay mixed kernels reliably.
KERNEL_LIBRARIES = {
    "libsparse_attention_head64_fused_hd576_kernel.so": os.path.join(
        EXTENSIONS_DIR,
        "simt",
        "sparse_attention_head64_fused_hd576.asc",
    ),
    "libsparse_attention_head64_hd576_kernel.so": os.path.join(
        EXTENSIONS_DIR,
        "simt",
        "sparse_attention_head64_hd576.asc",
    ),
    "libsparse_attention_postprocess_family_hd128_kernel.so": os.path.join(
        EXTENSIONS_DIR, "simt", "sparse_attention_postprocess_family_hd128.asc"
    ),
    "libsparse_attention_postprocess_family_hd512_kernel.so": os.path.join(
        EXTENSIONS_DIR, "simt", "sparse_attention_postprocess_family_hd512.asc"
    ),
    "libsparse_attention_query_pack_hd128_kernel.so": os.path.join(
        EXTENSIONS_DIR, "simt", "sparse_attention_query_pack_hd128.asc"
    ),
    "libsparse_attention_query_pack_hd512_kernel.so": os.path.join(
        EXTENSIONS_DIR, "simt", "sparse_attention_query_pack_hd512.asc"
    ),
    "libsparse_attention_score_family_hd128_kernel.so": os.path.join(
        EXTENSIONS_DIR, "simt", "sparse_attention_score_family_hd128.asc"
    ),
    "libsparse_attention_score_family_hd512_kernel.so": os.path.join(
        EXTENSIONS_DIR, "simt", "sparse_attention_score_family_hd512.asc"
    ),
}


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
    def initialize_options(self):
        super().initialize_options()
        self._kernel_outputs = []

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
        common_compile_cmd = [
            "bisheng",
            "-x",
            "asc",
            f"--npu-arch={NPU_ARCH}",
            "-shared",
            "-fPIC",
            "-std=c++17",
            opt_flag,
            f"-D_GLIBCXX_USE_CXX11_ABI={abi_value}",
        ]
        if debug_mode:
            common_compile_cmd.append("-g")

        include_flags = [f"-I{path}" for path in dep_paths["include_dirs"]]
        library_flags = [f"-L{path}" for path in dep_paths["library_dirs"]]
        dependency_flags = [
            "-ltorch_npu",
            "-ltorch_python",
            "-ltorch_cpu",
            "-ltorch",
            "-lc10",
            "-lm",
        ]
        kernel_output_dir = os.path.dirname(ext_fullpath)
        self._kernel_outputs.clear()

        try:
            for kernel_library, kernel_source in KERNEL_LIBRARIES.items():
                kernel_output = os.path.join(kernel_output_dir, kernel_library)
                self._build_kernel_library(
                    common_compile_cmd,
                    include_flags,
                    library_flags,
                    dependency_flags,
                    kernel_library,
                    kernel_source,
                    kernel_output,
                )
                self._kernel_outputs.append(kernel_output)

            compile_cmd = [
                *common_compile_cmd,
                *ext.sources,
                *include_flags,
                *library_flags,
                f"-L{kernel_output_dir}",
                *(f"-l:{name}" for name in KERNEL_LIBRARIES),
                "-Wl,-rpath,$ORIGIN",
                *dependency_flags,
                "-o",
                ext_fullpath,
            ]
            self.spawn(compile_cmd)
        except Exception as exc:
            raise CompileError(str(exc)) from exc

    def _build_kernel_library(
        self,
        common_compile_cmd,
        include_flags,
        library_flags,
        dependency_flags,
        kernel_library,
        kernel_source,
        kernel_output,
    ):
        compile_cmd = [
            *common_compile_cmd,
            kernel_source,
            *include_flags,
            *library_flags,
            f"-Wl,-soname,{kernel_library}",
            *dependency_flags,
            "-o",
            kernel_output,
        ]
        self.spawn(compile_cmd)

    def copy_extensions_to_source(self):
        super().copy_extensions_to_source()
        build_py = self.get_finalized_command("build_py")
        source_package_dir = build_py.get_package_dir(library_name)
        build_package_dir = os.path.join(self.build_lib, library_name)
        for kernel_library in KERNEL_LIBRARIES:
            self.copy_file(
                os.path.join(build_package_dir, kernel_library),
                os.path.join(source_package_dir, kernel_library),
                level=self.verbose,
            )

    def get_output_mapping(self):
        mapping = super().get_output_mapping()
        if self.inplace:
            build_py = self.get_finalized_command("build_py")
            source_package_dir = build_py.get_package_dir(library_name)
            build_package_dir = os.path.join(self.build_lib, library_name)
            for kernel_library in KERNEL_LIBRARIES:
                mapping[os.path.join(build_package_dir, kernel_library)] = (
                    os.path.join(source_package_dir, kernel_library)
                )
        return dict(sorted(mapping.items()))

    def get_outputs(self):
        outputs = super().get_outputs()
        outputs.extend(self._kernel_outputs)
        return list(dict.fromkeys(outputs))


def get_extensions():
    return [
        Extension(
            name=f"{library_name}._C",
            sources=HOST_SOURCES,
            language="asc",
        )
    ]


setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch", "torch_npu"],
    description="Ascend SIMT sparse attention migration helpers on torch_npu",
    cmdclass={"build_ext": AscendBuildExtension},
)
