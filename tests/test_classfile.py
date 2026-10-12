"""Synthetic class files exercise the loader independently of javac or CUDA."""

from pathlib import Path
import struct
import tempfile
import unittest

from jvmcuda.classfile import ClassFileError, UnsupportedFeature, load_method


def _u2(value):
    return struct.pack(">H", value)


def _u4(value):
    return struct.pack(">I", value)


def _i4(value):
    return struct.pack(">i", value)


class _Class:
    def __init__(self):
        self.pool = [None]
        self.this_class = self.add(7, _u2(self.utf8("Example")))
        self.super_class = self.add(7, _u2(self.utf8("java/lang/Object")))
        self.code_name = self.utf8("Code")
        self.flags = 0x0021
        self.major, self.minor = 65, 0
        self.interfaces = []
        self.methods = []
        self.attributes = []

    def add(self, tag, payload):
        index = len(self.pool)
        self.pool.append(bytes([tag]) + payload)
        if tag in (5, 6):
            self.pool.append(None)
        return index

    def utf8(self, value):
        if isinstance(value, str):
            units = value.encode("utf-16-be", "surrogatepass")
            encoded = bytearray()
            for i in range(0, len(units), 2):
                unit = int.from_bytes(units[i:i + 2], "big")
                if 1 <= unit <= 0x7F:
                    encoded.append(unit)
                elif unit <= 0x7FF:
                    encoded.extend((0xC0 | (unit >> 6), 0x80 | (unit & 0x3F)))
                else:
                    encoded.extend((0xE0 | (unit >> 12), 0x80 | ((unit >> 6) & 0x3F), 0x80 | (unit & 0x3F)))
            value = bytes(encoded)
        return self.add(1, _u2(len(value)) + value)

    @staticmethod
    def attribute(name_index, payload):
        return _u2(name_index) + _u4(len(payload)) + payload

    def method(self, code=b"\x04\xac", *, name="run", descriptor="()I", flags=0x0009,
               stack=1, locals=0, handlers=(), nested=(), attributes=()):
        name_index, descriptor_index = self.utf8(name), self.utf8(descriptor)
        attrs = list(attributes)
        if code is not None:
            body = _u2(stack) + _u2(locals) + _u4(len(code)) + code + _u2(len(handlers))
            body += b"".join(struct.pack(">HHHH", *handler) for handler in handlers)
            body += _u2(len(nested)) + b"".join(nested)
            attrs.insert(0, self.attribute(self.code_name, body))
        self.methods.append(_u2(flags) + _u2(name_index) + _u2(descriptor_index) + _u2(len(attrs)) + b"".join(attrs))

    def bytes(self):
        return (
            b"\xca\xfe\xba\xbe" + _u2(self.minor) + _u2(self.major)
            + _u2(len(self.pool)) + b"".join(entry for entry in self.pool if entry is not None)
            + _u2(self.flags) + _u2(self.this_class) + _u2(self.super_class)
            + _u2(len(self.interfaces)) + b"".join(map(_u2, self.interfaces))
            + _u2(0) + _u2(len(self.methods)) + b"".join(self.methods)
            + _u2(len(self.attributes)) + b"".join(self.attributes)
        )


class ClassFileTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "Example.class"

    def load(self, value, name="run", descriptor="()I"):
        self.path.write_bytes(value.bytes() if isinstance(value, _Class) else value)
        return load_method(self.path, name, descriptor)

    def method(self, code, **kwargs):
        value = _Class()
        value.method(code, **kwargs)
        return self.load(value, kwargs.get("name", "run"), kwargs.get("descriptor", "()I"))

    def test_all_prefix_truncations_are_rejected(self):
        value = _Class()
        value.method()
        data = value.bytes()
        for end in range(len(data)):
            with self.subTest(end=end), self.assertRaises(ClassFileError):
                self.load(data[:end])

    def test_bad_magic_and_trailing_bytes(self):
        value = _Class()
        value.method()
        data = value.bytes()
        for malformed in (b"oops" + data[4:], data + b"\0"):
            with self.subTest(data=malformed), self.assertRaises(ClassFileError):
                self.load(malformed)

    def test_supported_versions_and_preview_rejection(self):
        for major, minor in ((44, 0), (66, 0), (45, 4), (65, 65535), (52, 1)):
            value = _Class()
            value.method()
            value.major, value.minor = major, minor
            with self.subTest(major=major, minor=minor), self.assertRaises(UnsupportedFeature):
                self.load(value)

    def test_constant_pool_indices_and_signed_integers(self):
        value = _Class()
        negative = value.add(3, _u4(0x80000000))
        value.add(5, b"\xff" * 8)
        value.add(6, b"\0" * 8)
        value.add(4, _u4(0x7FC00000))
        for _ in range(260):
            value.utf8("unused")
        positive = value.add(3, _u4(2147483647))
        value.method(bytes([0x12, negative, 0x13]) + _u2(positive) + b"\x60\xac", stack=2)
        method = self.load(value)
        self.assertEqual(method.constants[negative], -2147483648)
        self.assertEqual(method.constants[positive], 2147483647)

    def test_unrelated_modern_constants_are_skipped(self):
        value = _Class()
        text = value.utf8("unused\0\U0001f600")
        value.add(8, _u2(text))
        name = value.utf8("helper")
        descriptor = value.utf8("()V")
        name_and_type = value.add(12, _u2(name) + _u2(descriptor))
        reference = value.add(10, _u2(value.this_class) + _u2(name_and_type))
        handle = value.add(15, b"\x06" + _u2(reference))
        value.add(16, _u2(descriptor))
        value.add(18, _u2(0) + _u2(name_and_type))
        integer_name_and_type = value.add(12, _u2(name) + _u2(value.utf8("I")))
        value.add(17, _u2(0) + _u2(integer_name_and_type))
        bootstrap = value.utf8("BootstrapMethods")
        value.attributes.append(value.attribute(bootstrap, _u2(1) + _u2(handle) + _u2(0)))
        value.method()
        self.assertEqual(self.load(value).code, b"\x04\xac")

    def test_modified_utf8_method_names_round_trip(self):
        for name in ("méthode", "\U00010400run", "\ud800run"):
            with self.subTest(name=repr(name)):
                self.assertEqual(self.method(b"\x04\xac", name=name).name, name)

    def test_invalid_modified_utf8_is_rejected(self):
        for encoded in (b"\0", b"\x80", b"\xc0\x81", b"\xc1\xbf", b"\xc2", b"\xe0\x80\x80", b"\xe1\x80", b"\xf0\x90\x80\x80"):
            value = _Class()
            value.utf8(encoded)
            value.method()
            with self.subTest(encoded=encoded), self.assertRaises(ClassFileError):
                self.load(value)

    def test_invalid_constant_pool_shapes(self):
        for tag, payload in ((99, b""), (7, _u2(0)), (7, _u2(65535)), (7, _u2(2)), (15, b"\0\0\1")):
            value = _Class()
            value.add(tag, payload)
            value.method()
            with self.subTest(tag=tag, payload=payload), self.assertRaises(ClassFileError):
                self.load(value)
        value = _Class()
        value.method()
        value.add(5, b"\0" * 8)
        value.pool.pop()  # Long must reserve a second pool slot.
        with self.assertRaises(ClassFileError):
            self.load(value)

    def test_ldc_requires_integer_constant(self):
        for index in (0, 1, 255):
            with self.subTest(index=index), self.assertRaises(ClassFileError):
                self.method(bytes([0x12, index, 0xAC]))
        value = _Class()
        index = value.add(4, _u4(0))
        value.method(b"\x13" + _u2(index) + b"\xac")
        with self.assertRaises(ClassFileError):
            self.load(value)

    def test_unknown_attributes_remain_bounded_and_ignored(self):
        value = _Class()
        name = value.utf8("VendorAttribute")
        attribute = value.attribute(name, b"opaque data")
        value.method(attributes=(attribute,), nested=(attribute,))
        value.attributes.append(attribute)
        value.attributes.append(_u2(name) + _u4(0xFFFFFFFF))
        with self.assertRaises(ClassFileError):
            self.load(value)

    def test_rejects_class_initialization_anywhere(self):
        for initializer_first in (True, False):
            value = _Class()
            if initializer_first:
                value.method(b"\xb1", name="<clinit>", descriptor="()V")
            value.method()
            if not initializer_first:
                value.method(b"\xb1", name="<clinit>", descriptor="()V")
            with self.subTest(first=initializer_first), self.assertRaises(UnsupportedFeature):
                self.load(value)

    def test_rejects_unsupported_class_structure(self):
        for feature in ("interface", "module", "implements", "superclass", "object"):
            value = _Class()
            value.method()
            if feature == "interface":
                value.flags |= 0x0200
            elif feature == "module":
                value.flags |= 0x8000
            elif feature == "implements":
                value.interfaces = [value.super_class]
            elif feature == "superclass":
                value.super_class = value.this_class
            else:
                value.super_class = 0
            with self.subTest(feature=feature), self.assertRaises(UnsupportedFeature):
                self.load(value)

    def test_selected_method_restrictions(self):
        for flags in (0x0001, 0x0029, 0x0109, 0x0409):
            value = _Class()
            value.method(None if flags & 0x0500 else b"\x04\xac", flags=flags)
            with self.subTest(flags=flags), self.assertRaises(UnsupportedFeature):
                self.load(value)
        for descriptor in ("()V", "(J)I", "([I)I", "(Ljava/lang/String;)I", "(I)J", "I", "(I"):
            with self.subTest(descriptor=descriptor), self.assertRaises(UnsupportedFeature):
                self.method(b"\x04\xac", descriptor=descriptor)
        with self.assertRaises(ClassFileError):
            self.method(b"\x04\xac", descriptor="(" + "I" * 256 + ")I", locals=256)
        with self.assertRaises(UnsupportedFeature):
            self.method(b"\x04\xac", handlers=((0, 1, 1, 0),))

    def test_missing_duplicate_and_invalid_code_attributes(self):
        value = _Class()
        value.method(None)
        with self.assertRaises(ClassFileError):
            self.load(value)
        value = _Class()
        value.method()
        value.method()
        with self.assertRaises(ClassFileError):
            self.load(value)
        value = _Class()
        body = _u2(1) + _u2(0) + _u4(2) + b"\x04\xac" + _u2(0) + _u2(0)
        value.method(attributes=(value.attribute(value.code_name, body),))
        with self.assertRaises(ClassFileError):
            self.load(value)
        for code in (b"", b"\0" * 65536):
            with self.subTest(length=len(code)), self.assertRaises(ClassFileError):
                self.method(code)
        value = _Class()
        value.method()
        with self.assertRaises(ClassFileError):
            self.load(value, "missing")

    def test_stack_underflow_and_overflow(self):
        for code in (b"\xac", b"\x57\x04\xac", b"\x59\xac", b"\x04\x5f\xac", b"\x04\x60\xac"):
            with self.subTest(code=code), self.assertRaisesRegex(ClassFileError, "underflow"):
                self.method(code, stack=2)
        with self.assertRaisesRegex(ClassFileError, "max_stack"):
            self.method(b"\x04\x59\xac", stack=1)
        with self.assertRaisesRegex(ClassFileError, "max_stack"):
            self.method(b"\x04\xac", stack=0)

    def test_local_initialization_and_bounds(self):
        for code, locals in ((b"\x1a\xac", 0), (b"\x04\x3c\x04\xac", 1), (b"\x15\x02\xac", 2)):
            with self.subTest(code=code), self.assertRaisesRegex(ClassFileError, "max_locals"):
                self.method(code, locals=locals)
        for code in (b"\x1a\xac", b"\x84\x00\x01\x1a\xac"):
            with self.subTest(code=code), self.assertRaisesRegex(ClassFileError, "Uninitialized"):
                self.method(code, locals=1)
        with self.assertRaisesRegex(ClassFileError, "Parameters"):
            self.method(b"\x04\xac", descriptor="(I)I", locals=0)

    def test_invalid_control_flow_targets(self):
        for code in (b"\xa7\0\x01", b"\xa7\xff\xff", b"\xa7\0\x03", b"\xc8\x7f\xff\xff\xff", b"\x03\x99\0\x02\x04\xac", b"\x04\xac\xa7\0\x01", b"\x04\xac\x99\0\x03"):
            with self.subTest(code=code), self.assertRaisesRegex(ClassFileError, "branch target"):
                self.method(code)
        with self.assertRaisesRegex(ClassFileError, "fallthrough"):
            self.method(b"\x04")

    def test_inconsistent_stack_depth_at_join(self):
        with self.assertRaisesRegex(ClassFileError, "Inconsistent stack"):
            self.method(b"\x1a\x99\0\x07\x04\xa7\0\x04\x00\x05\xac", descriptor="(I)I", locals=1, stack=2)

    def test_merged_uninitialized_local_reprocesses_join(self):
        # The initialized arm reaches the join first. The longer uninitialized
        # arm must intersect the prior state and revisit the already-seen load.
        code = b"\x1a\x99\0\x09\x00\x00\x00\xa7\0\x05\x04\x3c\x1b\xac"
        with self.assertRaisesRegex(ClassFileError, "Uninitialized"):
            self.method(code, descriptor="(I)I", locals=2)

    def test_wide_load_store_increment(self):
        code = b"\x04\xc4\x36\x01\x2c\xc4\x84\x01\x2c\x80\x00\xc4\x15\x01\x2c\xac"
        with self.assertRaisesRegex(ClassFileError, "max_locals"):
            self.method(code, locals=300)
        with self.assertRaisesRegex(ClassFileError, "Uninitialized"):
            self.method(b"\xc4\x84\x01\x2c\0\x01\x04\xac", locals=301)
        for nested in (0x16, 0xA9, 0x00):
            with self.subTest(nested=nested), self.assertRaises(UnsupportedFeature):
                self.method(bytes([0xC4, nested, 0, 0]))

    def test_invalid_switch_operands_and_targets(self):
        cases = [
            b"\x03\xaa\0\0" + _i4(0) + _i4(1) + _i4(0),
            b"\x03\xaa\0\0" + _i4(0) + _i4(-2147483648) + _i4(2147483647),
            b"\x03\xab\0\0" + _i4(0) + _i4(-1),
            b"\x03\xab\0\0" + _i4(0) + _i4(2147483647),
            b"\x03\xab\0\x01" + _i4(11) + _i4(0) + b"\x04\xac",
            b"\x03\xab\0\0" + _i4(2) + _i4(0) + b"\x04\xac",
            b"\x03\xaa\0\0" + _i4(19) + _i4(0) + _i4(0) + _i4(2) + b"\x04\xac",
        ]
        for keys in ((2, 1), (1, 1)):
            cases.append(b"\x03\xab\0\0" + _i4(27) + _i4(2) + _i4(keys[0]) + _i4(27) + _i4(keys[1]) + _i4(27) + b"\x04\xac")
        for code in cases:
            with self.subTest(code=code), self.assertRaises(ClassFileError):
                self.method(code)

    def test_dead_code_still_has_valid_shapes_without_dataflow(self):
        for dead in (b"\xb1", b"\x11\0", b"\x15\x02", b"\x12\0"):
            with self.subTest(dead=dead), self.assertRaises(ClassFileError):
                self.method(b"\x04\xac" + dead, locals=1)

    def test_truncated_operands_are_rejected(self):
        for code in (b"\x10", b"\x11\0", b"\x12", b"\x13\0", b"\x15", b"\x36", b"\x84\0", b"\x99\0", b"\xa7\0", b"\xc8\0\0\0", b"\xc4", b"\xc4\x15\0", b"\xc4\x84\0\0\0", b"\xaa", b"\xab\0\0\0"):
            with self.subTest(code=code), self.assertRaises(ClassFileError):
                self.method(code)

    def test_unsupported_opcodes_are_explicit(self):
        for opcode in (0x01, 0x09, 0x14, 0x58, 0x5A, 0xA8, 0xA9, 0xB1, 0xB2, 0xB8, 0xBC, 0xBE, 0xBF, 0xC2, 0xCA, 0xFF):
            with self.subTest(opcode=opcode), self.assertRaises(UnsupportedFeature):
                self.method(bytes([opcode]))


if __name__ == "__main__":
    unittest.main()
