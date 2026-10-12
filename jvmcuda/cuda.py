"""CUDA Driver/NVRTC backend. Java instructions execute only in interpreter.cu."""
from __future__ import annotations

import ctypes as c
import ctypes.util
from contextlib import contextmanager
from importlib.resources import files
import os
from pathlib import Path
import sys
import threading
from typing import Sequence

from .classfile import Method


class CudaError(RuntimeError):
    pass


class ExecutionError(RuntimeError):
    def __init__(self, invocation: int, pc: int, status: int):
        self.invocation = invocation
        self.pc = pc
        self.status = status
        reason = {1: "java.lang.ArithmeticException: / by zero",
                  2: "instruction budget exceeded",
                  3: "invalid interpreter state"}.get(status, f"unknown status {status}")
        super().__init__(f"invocation {invocation}, bytecode pc {pc}: {reason}")


_PTR = c.c_void_p
_DEVICE_PTR = c.c_uint64


class _Driver:
    def __init__(self):
        try:
            self.lib = c.CDLL("libcuda.so.1")
        except OSError as error:
            raise CudaError("Cannot load libcuda.so.1; use a Linux NVIDIA host with its driver installed") from error
        signatures = {
            "cuInit": [c.c_uint],
            "cuDeviceGet": [c.POINTER(c.c_int), c.c_int],
            "cuDeviceGetName": [_PTR, c.c_int, c.c_int],
            "cuDeviceGetAttribute": [c.POINTER(c.c_int), c.c_int, c.c_int],
            "cuDevicePrimaryCtxRetain": [c.POINTER(_PTR), c.c_int],
            "cuDevicePrimaryCtxRelease_v2": [c.c_int],
            "cuCtxPushCurrent_v2": [_PTR],
            "cuCtxPopCurrent_v2": [c.POINTER(_PTR)],
            "cuModuleLoadData": [c.POINTER(_PTR), _PTR],
            "cuModuleGetFunction": [c.POINTER(_PTR), _PTR, c.c_char_p],
            "cuModuleUnload": [_PTR],
            "cuMemAlloc_v2": [c.POINTER(_DEVICE_PTR), c.c_size_t],
            "cuMemFree_v2": [_DEVICE_PTR],
            "cuMemcpyHtoD_v2": [_DEVICE_PTR, _PTR, c.c_size_t],
            "cuMemcpyDtoH_v2": [_PTR, _DEVICE_PTR, c.c_size_t],
            "cuLaunchKernel": [_PTR, *([c.c_uint] * 7), _PTR, _PTR, _PTR],
            "cuCtxSynchronize": [],
            "cuGetErrorString": [c.c_int, c.POINTER(c.c_char_p)],
        }
        for name, signature in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes = signature
            function.restype = c.c_int
        self.call("cuInit", 0)

    def call(self, name: str, *arguments):
        result = getattr(self.lib, name)(*arguments)
        if result:
            message = c.c_char_p()
            self.lib.cuGetErrorString(result, c.byref(message))
            text = message.value.decode(errors="replace") if message.value else "unknown CUDA error"
            raise CudaError(f"{name}: {text} ({result})")


def _load_nvrtc():
    explicit = os.environ.get("JVMCUDA_NVRTC")
    if explicit:
        candidates = [explicit]
    else:
        candidates = [ctypes.util.find_library("nvrtc"), "libnvrtc.so.12", "libnvrtc.so.13"]
        for directory in sys.path:
            candidates.extend(str(path) for path in
                              (Path(directory) / "nvidia/cuda_nvrtc/lib").glob("libnvrtc.so*"))
    for candidate in filter(None, candidates):
        try:
            return c.CDLL(candidate)
        except OSError:
            continue
    raise CudaError("Cannot load NVRTC; install nvidia-cuda-nvrtc-cu12 or set JVMCUDA_NVRTC to its shared library")


def _compile(major: int, minor: int) -> bytes:
    nvrtc = _load_nvrtc()
    signatures = {
        "nvrtcCreateProgram": [c.POINTER(_PTR), c.c_char_p, c.c_char_p, c.c_int, _PTR, _PTR],
        "nvrtcCompileProgram": [_PTR, c.c_int, c.POINTER(c.c_char_p)],
        "nvrtcGetProgramLogSize": [_PTR, c.POINTER(c.c_size_t)],
        "nvrtcGetProgramLog": [_PTR, _PTR],
        "nvrtcGetPTXSize": [_PTR, c.POINTER(c.c_size_t)],
        "nvrtcGetPTX": [_PTR, _PTR],
        "nvrtcDestroyProgram": [c.POINTER(_PTR)],
    }
    for name, signature in signatures.items():
        fn = getattr(nvrtc, name)
        fn.argtypes = signature
        fn.restype = c.c_int

    def check(name, *args):
        result = getattr(nvrtc, name)(*args)
        if result:
            raise CudaError(f"{name} failed with NVRTC status {result}")

    source = files("jvmcuda").joinpath("interpreter.cu").read_bytes()
    program = _PTR()
    check("nvrtcCreateProgram", c.byref(program), source, b"interpreter.cu", 0, None, None)
    try:
        options = (c.c_char_p * 2)(b"--std=c++11", f"--gpu-architecture=compute_{major}{minor}".encode())
        status = nvrtc.nvrtcCompileProgram(program, len(options), options)
        if status:
            size = c.c_size_t()
            check("nvrtcGetProgramLogSize", program, c.byref(size))
            log = c.create_string_buffer(size.value)
            check("nvrtcGetProgramLog", program, log)
            raise CudaError(f"NVRTC compilation failed ({status}):\n{log.value.decode(errors='replace')}")
        size = c.c_size_t()
        check("nvrtcGetPTXSize", program, c.byref(size))
        ptx = c.create_string_buffer(size.value)
        check("nvrtcGetPTX", program, ptx)
        return ptx.raw
    finally:
        check("nvrtcDestroyProgram", c.byref(program))


class Executor:
    """A compiled GPU interpreter bound to one validated Method; use as a context manager.

    Each run launches one CUDA thread per argument row. Instances belong to their
    creating host thread. No Java computation is performed by the Python host.
    """
    def __init__(self, method: Method, device: int = 0):
        if not isinstance(device, int) or isinstance(device, bool) or not 0 <= device < 2**31:
            raise ValueError("device must be a nonnegative signed 32-bit integer")
        self.method = method
        self._thread = threading.get_ident()
        self._closed = True
        self._driver = _Driver()
        self._device = c.c_int()
        self._context = _PTR()
        self._module = _PTR()
        self._persistent: list[_DEVICE_PTR] = []
        self._driver.call("cuDeviceGet", c.byref(self._device), device)
        name = c.create_string_buffer(256)
        self._driver.call("cuDeviceGetName", name, len(name), self._device)
        self.device_name = name.value.decode()
        major, minor = c.c_int(), c.c_int()
        self._driver.call("cuDeviceGetAttribute", c.byref(major), 75, self._device)
        self._driver.call("cuDeviceGetAttribute", c.byref(minor), 76, self._device)
        ptx = c.create_string_buffer(_compile(major.value, minor.value))
        self._driver.call("cuDevicePrimaryCtxRetain", c.byref(self._context), self._device)
        self._closed = False
        try:
            with self._current():
                self._driver.call("cuModuleLoadData", c.byref(self._module), ptx)
                self._function = _PTR()
                self._driver.call("cuModuleGetFunction", c.byref(self._function), self._module, b"interpret")
                code = (c.c_ubyte * len(method.code)).from_buffer_copy(method.code)
                constants = (c.c_int * len(method.constants))(*method.constants)
                self._code = self._upload(code, self._persistent)
                self._constants = self._upload(constants, self._persistent)
        except BaseException:
            self.close()
            raise

    def _check_thread(self):
        if threading.get_ident() != self._thread:
            raise CudaError("Executor must be used and closed on its creating host thread")
        if self._closed:
            raise CudaError("Executor is closed")

    @contextmanager
    def _current(self):
        self._check_thread()
        self._driver.call("cuCtxPushCurrent_v2", self._context)
        try:
            yield
        finally:
            previous = _PTR()
            self._driver.call("cuCtxPopCurrent_v2", c.byref(previous))

    def _allocate(self, size: int, owned: list[_DEVICE_PTR]) -> _DEVICE_PTR:
        address = _DEVICE_PTR()
        self._driver.call("cuMemAlloc_v2", c.byref(address), max(size, 1))
        owned.append(address)
        return address

    def _upload(self, value, owned: list[_DEVICE_PTR]) -> _DEVICE_PTR:
        size = c.sizeof(value)
        address = self._allocate(size, owned)
        if size:
            self._driver.call("cuMemcpyHtoD_v2", address, value, size)
        return address

    def run(self, arguments: Sequence[Sequence[int]], *, step_limit: int = 1_000_000) -> list[int]:
        self._check_thread()
        if isinstance(step_limit, bool) or not isinstance(step_limit, int) or not 1 <= step_limit <= 2**64 - 1:
            raise ValueError("step_limit must be in [1, 2**64 - 1]")
        count = len(arguments)
        if count > 2**31 - 1:
            raise ValueError("too many invocations")
        method = self.method
        flat = []
        for index, row in enumerate(arguments):
            if len(row) != method.parameter_count:
                raise ValueError(f"invocation {index}: expected {method.parameter_count} integer arguments")
            for value in row:
                if isinstance(value, bool) or not isinstance(value, int) or not -(2**31) <= value < 2**31:
                    raise ValueError(f"invocation {index}: arguments must be signed 32-bit integers")
                flat.append(value)
        if not count:
            return []
        input_buffer = (c.c_int * len(flat))(*flat)
        results = (c.c_int * count)()
        statuses = (c.c_int * count)()
        error_pcs = (c.c_int * count)()
        owned: list[_DEVICE_PTR] = []
        with self._current():
            try:
                args_ptr = self._upload(input_buffer, owned)
                frame_size = count * (method.max_locals + method.max_stack) * 4
                frames_ptr = self._allocate(frame_size, owned)
                results_ptr = self._allocate(c.sizeof(results), owned)
                statuses_ptr = self._allocate(c.sizeof(statuses), owned)
                pcs_ptr = self._allocate(c.sizeof(error_pcs), owned)
                values = [self._code, c.c_int(len(method.code)), self._constants,
                          args_ptr, c.c_int(method.parameter_count), frames_ptr,
                          c.c_int(method.max_locals), c.c_int(method.max_stack),
                          results_ptr, statuses_ptr, pcs_ptr, c.c_ulonglong(step_limit), c.c_int(count)]
                parameters = (_PTR * len(values))(*(c.addressof(value) for value in values))
                self._driver.call("cuLaunchKernel", self._function, (count + 127) // 128, 1, 1,
                                  128, 1, 1, 0, None, parameters, None)
                self._driver.call("cuCtxSynchronize")
                self._driver.call("cuMemcpyDtoH_v2", statuses, statuses_ptr, c.sizeof(statuses))
                self._driver.call("cuMemcpyDtoH_v2", error_pcs, pcs_ptr, c.sizeof(error_pcs))
                for index, status in enumerate(statuses):
                    if status:
                        raise ExecutionError(index, error_pcs[index], status)
                self._driver.call("cuMemcpyDtoH_v2", results, results_ptr, c.sizeof(results))
                return list(results)
            finally:
                for address in reversed(owned):
                    self._driver.call("cuMemFree_v2", address)

    def close(self):
        if self._closed:
            return
        self._check_thread()
        try:
            with self._current():
                for address in reversed(self._persistent):
                    self._driver.call("cuMemFree_v2", address)
                self._persistent.clear()
                if self._module.value:
                    self._driver.call("cuModuleUnload", self._module)
        finally:
            self._closed = True
            self._driver.call("cuDevicePrimaryCtxRelease_v2", self._device)

    def __enter__(self):
        self._check_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
