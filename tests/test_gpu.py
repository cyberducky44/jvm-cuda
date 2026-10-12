"""Opt-in real GPU tests: JVMCUDA_GPU_TESTS=1, a JDK 21+, and NVRTC required."""
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest

from jvmcuda.classfile import load_method
from jvmcuda.cuda import ExecutionError, Executor
from tests.test_classfile import _Class, _i4

ROOT = Path(__file__).resolve().parents[1]
MIN_INT = -(2**31)
MAX_INT = 2**31 - 1


@unittest.skipUnless(os.environ.get("JVMCUDA_GPU_TESTS") == "1", "set JVMCUDA_GPU_TESTS=1 for real CUDA execution")
class GpuSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        java_home = os.environ.get("JAVA_HOME")
        cls.java = str(Path(java_home) / "bin/java") if java_home else shutil.which("java")
        javac = str(Path(java_home) / "bin/javac") if java_home else shutil.which("javac")
        if not cls.java or not javac:
            raise RuntimeError("GPU differential tests require java and javac")
        cls.directory = tempfile.TemporaryDirectory(prefix="jvmcuda-tests-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.classes = Path(cls.directory.name)
        wide = cls.classes / "Wide.java"
        # Cross both the 8-bit local index and the 8-bit constant pool index boundaries.
        declarations = "\n".join(f"int v{i} = x + {100000 + i};" for i in range(270))
        wide.write_text("public class Wide { public static int run(int x) {\n" + declarations
                        + "\nv269 += 300; return v0 + v269; }}\n")
        distant = cls.classes / "Distant.java"
        distant.write_text("public class Distant { public static int run(int x) { if (x < 0) {\n"
                           + "x += 1;\n" * 11000 + "} return x; }}\n")
        subprocess.run([javac, "--release", "21", "-d", str(cls.classes),
                        str(ROOT / "examples/IntegerMath.java"),
                        str(ROOT / "tests/java/Reference.java"), str(wide), str(distant)],
                       check=True, capture_output=True, text=True, timeout=60)

    def reference(self, name, rows, class_name="IntegerMath"):
        count = len(rows[0])
        process = subprocess.run([self.java, "-cp", str(self.classes), "Reference", class_name, name, str(count)],
                                 input="".join(" ".join(map(str, row)) + "\n" for row in rows),
                                 capture_output=True, text=True, check=True, timeout=30)
        return process.stdout.splitlines()

    def compare(self, name, rows, class_name="IntegerMath"):
        descriptor = "(" + "I" * len(rows[0]) + ")I"
        method = load_method(self.classes / f"{class_name}.class", name, descriptor)
        with Executor(method) as executor:
            actual = executor.run(rows)
        self.assertEqual(actual, [int(value) for value in self.reference(name, rows, class_name)])

    def test_wrapping_arithmetic_and_batch_isolation(self):
        edge = [MIN_INT, MIN_INT + 1, -1, 0, 1, MAX_INT]
        rows = [[a, b] for a in edge for b in edge]
        rng = random.Random(1923)
        rows.extend([[rng.randint(MIN_INT, MAX_INT), rng.randint(MIN_INT, MAX_INT)] for _ in range(513)])
        self.compare("mix", rows)
        self.compare("negate", [[value] for value in edge])

    def test_division_and_remainder_signs_and_overflow(self):
        rows = [[a, b] for a in [MIN_INT, MIN_INT + 1, -7, -1, 0, 1, 7, MAX_INT]
                for b in [MIN_INT, -3, -1, 1, 3, MAX_INT]]
        for name in ["divide", "remainder"]:
            with self.subTest(operation=name):
                self.compare(name, rows)

    def test_division_by_zero_reports_faulting_invocation(self):
        for name in ["divide", "remainder"]:
            with self.subTest(operation=name):
                method = load_method(self.classes / "IntegerMath.class", name, "(II)I")
                self.assertEqual(self.reference(name, [[9, 0]]), ["java.lang.ArithmeticException"])
                with Executor(method) as executor:
                    with self.assertRaises(ExecutionError) as caught:
                        executor.run([[9, 3], [9, 0], [12, 2]])
                    self.assertEqual(caught.exception.status, 1)
                    self.assertEqual(caught.exception.invocation, 1)
                    self.assertEqual(method.code[caught.exception.pc], 0x6c if name == "divide" else 0x70)
                    # A Java arithmetic failure does not poison subsequent GPU executions.
                    self.assertEqual(executor.run([[12, 3]]), [4 if name == "divide" else 0])

    def test_shift_masks_and_sign_extension(self):
        rows = [[value, shift] for value in [MIN_INT, -123, -1, 0, 1, MAX_INT]
                for shift in [MIN_INT, -33, -32, -1, 0, 1, 31, 32, 33, MAX_INT]]
        for name in ["left", "right", "unsignedRight"]:
            with self.subTest(operation=name):
                self.compare(name, rows)

    def test_signed_comparisons(self):
        edge = [MIN_INT, -1, 0, 1, MAX_INT]
        self.compare("compare", [[a, b] for a in edge for b in edge])

    def test_loop_and_signed_wide_increment(self):
        self.compare("sum", [[-1], [0], [1], [10], [1000], [100000]])
        self.compare("increment", [[MIN_INT], [-1], [0], [MAX_INT]])

    def test_switches_with_negative_keys_and_defaults(self):
        self.compare("denseSwitch", [[x] for x in [MIN_INT, -3, -2, -1, 0, 1, 2, MAX_INT]])
        self.compare("sparseSwitch", [[x] for x in [MIN_INT, -2000000000, -1, 17, 18, 1000000000, MAX_INT]])

    def test_zero_arguments_and_integer_constant(self):
        self.compare("constant", [[], [], []])

    def test_wide_local_and_constant_pool_indexes(self):
        self.compare("run", [[MIN_INT], [-1], [0], [MAX_INT]], "Wide")

    def test_long_branch_displacements(self):
        self.compare("run", [[MIN_INT], [-11001], [-1], [0], [MAX_INT]], "Distant")

    def test_instruction_budget_terminates_infinite_loop(self):
        method = load_method(self.classes / "IntegerMath.class", "spin", "(I)I")
        with Executor(method) as executor:
            with self.assertRaises(ExecutionError) as caught:
                executor.run([[MAX_INT]], step_limit=100)
            self.assertEqual(caught.exception.status, 2)
            self.assertEqual(caught.exception.invocation, 0)

    def test_stack_manipulation_and_frame_return(self):
        cases = [
            (b"\x00\x10\x80\x59\x11\x80\x00\x5f\x57\x60\x74\xac", 3, 0, 32896),
            (b"\x04\x05\xac", 2, 0, 2),
            (b"\x04\xc4\x36\x01\x2c\xc4\x84\x01\x2c\x80\x00\xc4\x15\x01\x2c\xac", 1, 301, -32767),
        ]
        for code, stack, locals_count, expected in cases:
            with self.subTest(code=code):
                value = _Class()
                value.major = 49  # JVM infers frames; no StackMapTable needed.
                value.method(code, stack=stack, locals=locals_count)
                (self.classes / "Example.class").write_bytes(value.bytes())
                self.assertEqual(self.reference("run", [[]], "Example"), [str(expected)])
                self.compare("run", [[]], "Example")

    def test_default_only_switch_at_each_alignment(self):
        for prefix in range(4):
            with self.subTest(alignment=prefix):
                pc = prefix + 1
                padding = (-(pc + 1)) % 4
                code = b"\0" * prefix + b"\x03\xab" + b"\0" * padding
                code += _i4(1 + padding + 8) + _i4(0) + b"\x04\xac"
                value = _Class()
                value.major = 49
                value.method(code)
                (self.classes / "Example.class").write_bytes(value.bytes())
                self.compare("run", [[]], "Example")

    def test_definite_assignment_at_control_flow_join(self):
        value = _Class()
        value.major = 49
        value.method(b"\x1a\x99\0\x08\x04\x3c\xa7\0\x05\x05\x3c\x1b\xac",
                     descriptor="(I)I", locals=2)
        (self.classes / "Example.class").write_bytes(value.bytes())
        self.compare("run", [[0], [-1], [1]], "Example")

    def test_argument_boundaries_are_not_silently_truncated(self):
        method = load_method(self.classes / "IntegerMath.class", "negate", "(I)I")
        with Executor(method) as executor:
            for rows in [[[MAX_INT + 1]], [[MIN_INT - 1]], [[True]], [[]], [[1, 2]]]:
                with self.subTest(arguments=rows), self.assertRaises(ValueError):
                    executor.run(rows)
            for budget in [0, -1, 2**64, True]:
                with self.subTest(budget=budget), self.assertRaises(ValueError):
                    executor.run([[1]], step_limit=budget)
            self.assertEqual(executor.run([[MIN_INT], [MAX_INT]]), [MIN_INT, -MAX_INT])


if __name__ == "__main__":
    unittest.main()
