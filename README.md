# JVM on CUDA

An experimental direct GPU bytecode interpreter. The long-term target is a JVM
runtime port to CUDA. **This repository does not yet implement a complete JVM or
run arbitrary Java applications.** The first verified increment executes selected
static integer methods from real Java class files on an NVIDIA GPU.

There is no CPU bytecode interpreter or execution fallback. Python loads and
validates the method, NVRTC compiles the CUDA interpreter to PTX, and the CUDA
Driver API launches it. Each CUDA thread executes one independent invocation;
these invocations are not Java threads and share no Java objects.

## Requirements

- Linux, an NVIDIA GPU, and a compatible installed NVIDIA driver.
- Python 3.11 or newer.
- CUDA NVRTC and its companion builtins library. `nvcc` and a host C++ compiler
  are not required.
- A JDK capable of compiling with `--release 21` to build examples and run the
  differential tests. Executing an existing supported class file needs no JDK.

On a machine with Python packaging tools available, install from this directory:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install '.[cuda]'
```

The optional `cuda` dependency installs NVIDIA's NVRTC package, not a GPU driver.
With an existing NVRTC installation, set `JVMCUDA_NVRTC` to the full path of
`libnvrtc.so.12` or `libnvrtc.so.13` if the library is not discoverable. Keep the
companion builtins library from the same NVRTC distribution alongside it.

Source-checkout execution via `python3 -m jvmcuda` needs no Python dependencies
beyond the standard library when the driver and NVRTC libraries are available.

## Run a Java method

From the repository root:

```sh
mkdir -p build
javac --release 21 -d build examples/IntegerMath.java
python3 -m jvmcuda build/IntegerMath.class sum 1000
```

The result is `499500` on stdout. The selected GPU name is reported on stderr.
The runner infers an integer-only method descriptor from the argument count.
Use `--descriptor '(II)I'` to select a descriptor explicitly, `--device 1` to
select another CUDA device, or `--steps 2000000` to change the per-invocation
instruction budget. The default budget is 1,000,000 instructions, not a timeout.

Input values must be signed 32-bit integers. A loading, validation, CUDA, or
execution error produces a diagnostic on stderr and exit status 1, without
printing a successful result.

The Python API supports batches:

```python
from jvmcuda.classfile import load_method
from jvmcuda.cuda import Executor

method = load_method("build/IntegerMath.class", "mix", "(II)I")
with Executor(method) as gpu:
    results = gpu.run([[10, 20], [-1, 7], [2147483647, 1]])
    print(results)
```

Construct methods through `load_method`; the executor relies on its validation.
An executor is bound to one method and its creating host thread. Its context
manager releases CUDA allocations, the compiled module, and its retained device
context. Device context pushes/pops preserve the host thread's previous CUDA
context. A batch execution error raises `ExecutionError`, identifying the first
failing invocation in input order and its bytecode offset. Partial batch results
are not returned.

## Supported boundary

Class-file versions 45 through 65 are accepted (through Java 21), excluding
preview versions. Only the selected method is bytecode-verified.

The selected method must:

- Be declared static, concrete, non-native, and unsynchronized.
- Accept only `int` parameters and return `int`: descriptors of the form `(I*)I`.
- Have no exception handlers and use only the instructions listed below.
- Belong to a class directly extending `java.lang.Object`, implementing no
  interfaces, and defining no `<clinit>` class initializer.

| Instruction family | Supported instructions |
| --- | --- |
| Constants | `iconst_m1` through `iconst_5`, `bipush`, `sipush`, integer `ldc` and `ldc_w` |
| Locals | `iload`, `iload_0` through `iload_3`, `istore`, `istore_0` through `istore_3`, `iinc`; `wide` forms of `iload`, `istore`, `iinc` |
| Stack | `nop`, `pop`, `dup`, `swap` |
| Arithmetic | `iadd`, `isub`, `imul`, `idiv`, `irem`, `ineg` |
| Bits | `ishl`, `ishr`, `iushr`, `iand`, `ior`, `ixor` |
| Control flow | All integer-zero and integer-pair comparisons, `goto`, `goto_w`, `tableswitch`, `lookupswitch`, `ireturn` |

Integer overflow wraps to 32 bits. Division truncates toward zero; remainder
follows Java's sign rules. `Integer.MIN_VALUE / -1` returns `Integer.MIN_VALUE`.
Shift distances are masked to five bits. Divide/remainder by zero is detected on
the GPU and reported as an execution error identifying
`java.lang.ArithmeticException`; Java exception construction and unwinding are
not implemented. Budget exhaustion also produces an execution error.

The loader checks bounded class-file structures, modified UTF-8, integer constant
references, instruction boundaries, branch/switch targets, reachable stack depths,
and definite assignment of locals across control-flow joins. The kernel adds
code/frame bounds checks. **This is not whole-class JVM verification or a security
sandbox.** Use trusted input; instruction budgets do not bound host parsing time,
allocation sizes, or GPU wall-clock time.

Not implemented: objects or arrays, fields, method invocation, Java libraries,
other primitive types and conversions, exception handling, class initialization,
inheritance/interfaces, Java threads and monitors, garbage collection, JNI,
reflection, or application/JAR launching. Unsupported instructions are rejected
before launch rather than emulated on the CPU. Unselected methods may contain
unsupported bytecode, but they cannot be invoked by the GPU method.

## Verification

Host-only loader regressions, requiring neither a JDK nor a GPU:

```sh
python3 -m unittest discover -s tests -p test_classfile.py -v
```

Full suite with real GPU execution and OpenJDK differential comparisons:

```sh
export JAVA_HOME=/path/to/jdk-21
export JVMCUDA_GPU_TESTS=1
# Set JVMCUDA_NVRTC here if required by your installation.
python3 -m unittest discover -s tests -v
```

GPU tests skip unless explicitly enabled. Once enabled, missing CUDA/JDK/NVRTC
prerequisites are failures, not silent skips. Tests compile actual Java fixtures
and compare GPU outputs against separate OpenJDK processes. Synthetic class files
cover bytecode boundaries difficult to express in Java source.

Verified on Debian 13, NVIDIA RTX 3070 (compute capability 8.6), driver 595.58.03,
NVRTC 12.9.86, and OpenJDK 21.0.12.1: **39 tests passed**, including **15 real-GPU
tests**. The command-line `sum(1000)` smoke run also returned the expected result
on that GPU. Coverage includes overflow, division/remainder signs and zero,
masked shifts, signed comparisons, loops, switches at different alignments,
wide locals/constants, long branch offsets, batch isolation, and execution budgets.

No speedup or full JVM compatibility is claimed. The present interpreter uses
per-invocation frames in device memory and incurs compilation and transfer costs.
