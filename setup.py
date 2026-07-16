import contextlib
import os
import platform
import re
import shutil
import sys
from collections.abc import Generator
from pathlib import Path

import setuptools
from setuptools.command import build_ext

free_threaded = "experimental free-threading build" in sys.version
py_limited_api = sys.version_info >= (3, 12) and not free_threaded
options = {"bdist_wheel": {"py_limited_api": "cp312"}} if py_limited_api else {}


def is_cibuildwheel() -> bool:
    return os.getenv("CIBUILDWHEEL") is not None


@contextlib.contextmanager
def _maybe_patch_toolchains() -> Generator[None, None, None]:
    def format_toolchain_arguments(match: re.Match) -> str:
        suffix = "ignore_root_user_error = True"
        arguments = match.group(1)
        if arguments.endswith("\n"):
            arguments += f"    {suffix},\n"
        else:
            arguments += f", {suffix}"
        return f"python.toolchain({arguments})"

    module_path = Path("MODULE.bazel")
    contents = module_path.read_text()
    patch = is_cibuildwheel() and platform.system() == "Linux"
    try:
        if patch:
            module_path.write_text(
                re.sub(
                    r"python.toolchain\(([\w\"\s,.=]*)\)",
                    format_toolchain_arguments,
                    contents,
                )
            )
        yield
    finally:
        if patch:
            module_path.write_text(contents)


class BazelExtension(setuptools.Extension):
    def __init__(self, name: str, bazel_target: str, **kwargs):
        super().__init__(name=name, sources=[], **kwargs)
        self.free_threaded = free_threaded
        self.bazel_target = bazel_target


class BuildBazelExtension(build_ext.build_ext):
    def run(self):
        for extension in self.extensions:
            self.bazel_build(extension)
        self.spawn(["bazel", "shutdown"])

    def copy_extensions_to_source(self):
        pass

    def bazel_build(self, extension: BazelExtension) -> None:
        temporary_path = Path(self.build_temp)
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        arguments = [
            "bazel",
            "run",
            extension.bazel_target,
            f"--symlink_prefix={temporary_path / 'bazel-'}",
            f"--target_python_version={python_version}",
        ]
        if self.debug:
            arguments.append("--config=debug")
        if extension.py_limited_api:
            arguments.append("--py_limited_api=cp312")
        if extension.free_threaded:
            arguments.append("--free_threaded=yes")

        with _maybe_patch_toolchains():
            self.spawn(arguments)

        if platform.system() == "Windows":
            suffix = ".pyd"
        else:
            suffix = ".abi3.so" if extension.py_limited_api else ".so"

        source_directory = temporary_path / "bazel-bin" / "src"
        destination_directory = Path(self.build_lib) / "sip_qp_python"
        for root, directories, files in os.walk(source_directory, topdown=True):
            directories[:] = [d for d in directories if "runfiles" not in d]
            for filename in files:
                path = Path(filename)
                exact_suffix = "".join(path.suffixes)
                should_copy = exact_suffix == suffix or path.suffix == ".pyi"
                should_copy |= Path(root) == source_directory and filename == "py.typed"
                if should_copy:
                    relative = os.path.relpath(root, source_directory)
                    output_directory = destination_directory / relative
                    output_directory.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(Path(root) / path, output_directory / path)


setuptools.setup(
    cmdclass={"build_ext": BuildBazelExtension},
    package_data={"sip_qp_python": ["py.typed", "*.pyi", "**/*.pyi"]},
    ext_modules=[
        BazelExtension(
            name="sip_qp_python.sip_qp_python_ext",
            bazel_target="//src:sip_qp_python_ext_stubgen",
            free_threaded=free_threaded,
            py_limited_api=py_limited_api,
        )
    ],
    options=options,
)
