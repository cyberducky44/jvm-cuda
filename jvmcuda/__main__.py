"""Run a validated static integer Java method on an NVIDIA GPU."""
import argparse
import sys

from .classfile import ClassFileError, load_method
from .cuda import CudaError, ExecutionError, Executor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("class_file", help="path to a Java .class file (through Java 21)")
    parser.add_argument("method", help="declared static method name")
    parser.add_argument("arguments", nargs="*", type=int, help="signed 32-bit integer arguments")
    parser.add_argument("--descriptor", help="JVM descriptor; default: infer (I...)I from argument count")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ordinal")
    parser.add_argument("--steps", type=int, default=1_000_000, help="instruction budget per invocation")
    args = parser.parse_args()
    descriptor = args.descriptor or "(" + "I" * len(args.arguments) + ")I"
    try:
        method = load_method(args.class_file, args.method, descriptor)
        with Executor(method, device=args.device) as executor:
            result = executor.run([args.arguments], step_limit=args.steps)[0]
            print(f"CUDA device: {executor.device_name}", file=sys.stderr)
            print(result)
        return 0
    except (OSError, ValueError, ClassFileError, CudaError, ExecutionError) as error:
        print(f"jvmcuda: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
