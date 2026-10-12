"""Bounded class-file loading and verification for the integer GPU subset.

Only the selected method's bytecode is verified. This is not a whole-class JVM
verifier or a security sandbox; unrelated attributes and method bodies are skipped.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path


class ClassFileError(ValueError):
    """Malformed class file or selected method."""


class UnsupportedFeature(ClassFileError):
    """A valid-looking feature outside the supported execution subset."""


@dataclass(frozen=True)
class Method:
    name: str
    descriptor: str
    code: bytes
    max_stack: int
    max_locals: int
    parameter_count: int
    constants: tuple[int, ...]


class _Reader:
    def __init__(self, data: bytes | memoryview):
        self.data = memoryview(data)
        self.pos = 0

    def take(self, size: int) -> memoryview:
        if size < 0 or size > len(self.data) - self.pos:
            raise ClassFileError(f"Truncated class file at offset {self.pos}")
        start = self.pos
        self.pos += size
        return self.data[start:self.pos]

    def uint(self, size: int) -> int:
        return int.from_bytes(self.take(size), "big")

    def u1(self) -> int:
        return self.uint(1)

    def u2(self) -> int:
        return self.uint(2)

    def u4(self) -> int:
        return self.uint(4)

    def finish(self) -> None:
        if self.pos != len(self.data):
            raise ClassFileError("Trailing bytes in class-file structure")


def _modified_utf8(data: memoryview) -> str:
    # CONSTANT_Utf8 uses UTF-16 code units encoded separately, not UTF-8.
    units: list[int] = []
    i = 0
    while i < len(data):
        first = data[i]
        i += 1
        if 1 <= first <= 0x7F:
            units.append(first)
        elif 0xC0 <= first <= 0xDF:
            if i == len(data) or data[i] & 0xC0 != 0x80:
                raise ClassFileError("Invalid modified UTF-8 constant")
            value = ((first & 0x1F) << 6) | (data[i] & 0x3F)
            i += 1
            if value < 0x80 and not (first == 0xC0 and value == 0):
                raise ClassFileError("Overlong modified UTF-8 constant")
            units.append(value)
        elif 0xE0 <= first <= 0xEF:
            if i + 1 >= len(data) or any(x & 0xC0 != 0x80 for x in data[i:i + 2]):
                raise ClassFileError("Invalid modified UTF-8 constant")
            value = ((first & 0x0F) << 12) | ((data[i] & 0x3F) << 6) | (data[i + 1] & 0x3F)
            i += 2
            if value < 0x800:
                raise ClassFileError("Overlong modified UTF-8 constant")
            units.append(value)
        else:
            raise ClassFileError("Invalid modified UTF-8 constant")
    chars: list[str] = []
    i = 0
    while i < len(units):
        value = units[i]
        i += 1
        if 0xD800 <= value <= 0xDBFF and i < len(units) and 0xDC00 <= units[i] <= 0xDFFF:
            value = 0x10000 + ((value - 0xD800) << 10) + units[i] - 0xDC00
            i += 1
        chars.append(chr(value))
    return "".join(chars)


class _Pool:
    def __init__(self, reader: _Reader, major: int):
        count = reader.u2()
        if count == 0:
            raise ClassFileError("Empty constant pool")
        self.entries: list[tuple[int, object] | None] = [None] * count
        self.constants = [0] * count
        index = 1
        while index < count:
            tag = reader.u1()
            if tag == 1:
                value = _modified_utf8(reader.take(reader.u2()))
            elif tag in (3, 4):
                value = reader.u4()
                if tag == 3:
                    self.constants[index] = value if value < 0x80000000 else value - 0x100000000
            elif tag in (5, 6):
                value = reader.uint(8)
                if index + 1 >= count:
                    raise ClassFileError("Two-slot constant at end of constant pool")
            elif tag in (7, 8, 16, 19, 20):
                value = reader.u2()
            elif tag in (9, 10, 11, 12, 17, 18):
                value = (reader.u2(), reader.u2())
            elif tag == 15:
                value = (reader.u1(), reader.u2())
            else:
                raise ClassFileError(f"Unknown constant-pool tag {tag}")
            minimum = {15: 51, 16: 51, 18: 51, 17: 55, 19: 53, 20: 53}.get(tag, 45)
            if major < minimum:
                raise ClassFileError(f"Constant-pool tag {tag} requires class version {minimum}")
            self.entries[index] = (tag, value)
            index += 2 if tag in (5, 6) else 1
        for entry in self.entries[1:]:
            if entry is None:
                continue
            tag, value = entry
            if tag in (7, 8, 16, 19, 20):
                self.get(value, 1)
            elif tag in (9, 10, 11):
                self.get(value[0], 7)
                self.get(value[1], 12)
            elif tag == 12:
                self.get(value[0], 1)
                self.get(value[1], 1)
            elif tag in (17, 18):
                self.get(value[1], 12)
            elif tag == 15:
                kind, target = value
                if kind in (1, 2, 3, 4):
                    expected = (9,)
                elif kind in (5, 8):
                    expected = (10,)
                elif kind in (6, 7):
                    expected = (10, 11) if major >= 52 else (10,)
                elif kind == 9:
                    expected = (11,)
                else:
                    raise ClassFileError("Invalid method-handle reference kind")
                self.get(target, *expected)

    def get(self, index: int, *tags: int):
        if not 0 < index < len(self.entries):
            raise ClassFileError(f"Invalid constant-pool index {index}")
        entry = self.entries[index]
        if entry is None or entry[0] not in tags:
            raise ClassFileError(f"Wrong constant-pool type at index {index}")
        return entry[1]

    def text(self, index: int) -> str:
        return self.get(index, 1)


def _attributes(reader: _Reader, pool: _Pool):
    for _ in range(reader.u2()):
        name = pool.text(reader.u2())
        yield name, _Reader(reader.take(reader.u4()))


def _code_attribute(reader: _Reader, pool: _Pool):
    max_stack, max_locals = reader.u2(), reader.u2()
    length = reader.u4()
    if not 0 < length <= 65535:
        raise ClassFileError("Code length must be between 1 and 65535")
    code = bytes(reader.take(length))
    handlers = reader.u2()
    for _ in range(handlers):
        start, end, target, catch = reader.u2(), reader.u2(), reader.u2(), reader.u2()
        if not 0 <= start < end <= length or target >= length:
            raise ClassFileError("Invalid exception-table range")
        if catch:
            pool.get(catch, 7)
    for _name, _attribute in _attributes(reader, pool):
        pass
    reader.finish()
    return code, max_stack, max_locals, handlers


@dataclass(frozen=True)
class _Instruction:
    pops: int
    pushes: int
    successors: tuple[int, ...]
    load: int | None = None
    store: int | None = None


def _verify(code: bytes, max_stack: int, max_locals: int, parameters: int, pool: _Pool) -> None:
    if parameters > max_locals:
        raise ClassFileError("Parameters exceed max_locals")
    instructions: dict[int, _Instruction] = {}
    explicit_edges: list[tuple[int, int]] = []
    reader = _Reader(code)
    binary = {0x60, 0x64, 0x68, 0x6C, 0x70, 0x78, 0x7A, 0x7C, 0x7E, 0x80, 0x82}

    def signed(size: int) -> int:
        return int.from_bytes(reader.take(size), "big", signed=True)

    while reader.pos < len(code):
        pc = reader.pos
        opcode = reader.u1()
        pops = pushes = 0
        load = store = None
        targets: tuple[int, ...] = ()
        falls_through = True
        if opcode == 0x00:
            pass
        elif 0x02 <= opcode <= 0x08:
            pushes = 1
        elif opcode in (0x10, 0x11):
            signed(1 if opcode == 0x10 else 2)
            pushes = 1
        elif opcode in (0x12, 0x13):
            pool.get(reader.uint(1 if opcode == 0x12 else 2), 3)
            pushes = 1
        elif opcode == 0x15 or 0x1A <= opcode <= 0x1D:
            load = reader.u1() if opcode == 0x15 else opcode - 0x1A
            pushes = 1
        elif opcode == 0x36 or 0x3B <= opcode <= 0x3E:
            store = reader.u1() if opcode == 0x36 else opcode - 0x3B
            pops = 1
        elif opcode == 0x57:
            pops = 1
        elif opcode == 0x59:
            pops, pushes = 1, 2
        elif opcode == 0x5F:
            pops, pushes = 2, 2
        elif opcode in binary:
            pops, pushes = 2, 1
        elif opcode == 0x74:
            pops = pushes = 1
        elif opcode == 0x84:
            load = store = reader.u1()
            signed(1)
        elif 0x99 <= opcode <= 0xA4:
            pops = 1 if opcode <= 0x9E else 2
            targets = (pc + signed(2),)
        elif opcode in (0xA7, 0xC8):
            targets = (pc + signed(2 if opcode == 0xA7 else 4),)
            falls_through = False
        elif opcode in (0xAA, 0xAB):
            padding = reader.take((-reader.pos) % 4)
            if any(padding):
                raise ClassFileError(f"Nonzero switch padding at bytecode offset {pc}")
            default = pc + signed(4)
            if opcode == 0xAA:
                low, high = signed(4), signed(4)
                count = high - low + 1
                if count <= 0 or count > (len(code) - reader.pos) // 4:
                    raise ClassFileError(f"Invalid tableswitch bounds at bytecode offset {pc}")
                targets = (default,) + tuple(pc + signed(4) for _ in range(count))
            else:
                count = signed(4)
                if count < 0 or count > (len(code) - reader.pos) // 8:
                    raise ClassFileError(f"Invalid lookupswitch size at bytecode offset {pc}")
                offsets = [default]
                previous = None
                for _ in range(count):
                    key, offset = signed(4), signed(4)
                    if previous is not None and key <= previous:
                        raise ClassFileError(f"Unsorted lookupswitch keys at bytecode offset {pc}")
                    previous = key
                    offsets.append(pc + offset)
                targets = tuple(offsets)
            pops = 1
            falls_through = False
        elif opcode == 0xAC:
            pops = 1
            falls_through = False
        elif opcode == 0xC4:
            nested = reader.u1()
            if nested not in (0x15, 0x36, 0x84):
                raise UnsupportedFeature(f"Unsupported wide opcode 0x{nested:02x} at {pc}")
            index = reader.u2()
            if nested == 0x15:
                load, pushes = index, 1
            elif nested == 0x36:
                store, pops = index, 1
            else:
                load = store = index
                signed(2)
        else:
            raise UnsupportedFeature(f"Unsupported opcode 0x{opcode:02x} at bytecode offset {pc}")
        if any(index is not None and index >= max_locals for index in (load, store)):
            raise ClassFileError(f"Local index exceeds max_locals at bytecode offset {pc}")
        explicit_edges.extend((pc, target) for target in targets)
        if falls_through:
            targets += (reader.pos,)
        instructions[pc] = _Instruction(pops, pushes, targets, load, store)

    # Every explicit target must be an instruction boundary, even in dead code.
    # A fallthrough to code_length is diagnosed only if it is reachable.
    for pc, target in explicit_edges:
        if target not in instructions:
            raise ClassFileError(f"Invalid branch target {target} at bytecode offset {pc}")

    states = {0: (0, (1 << parameters) - 1)}
    pending = deque([0])
    queued = {0}
    while pending:
        pc = pending.popleft()
        queued.remove(pc)
        depth, initialized = states[pc]
        instruction = instructions[pc]
        if depth < instruction.pops:
            raise ClassFileError(f"Operand-stack underflow at bytecode offset {pc}")
        depth += instruction.pushes - instruction.pops
        if depth > max_stack:
            raise ClassFileError(f"Operand stack exceeds max_stack at bytecode offset {pc}")
        if instruction.load is not None and not initialized & (1 << instruction.load):
            raise ClassFileError(f"Uninitialized local at bytecode offset {pc}")
        if instruction.store is not None:
            initialized |= 1 << instruction.store
        for target in instruction.successors:
            if target == len(code):
                raise ClassFileError(f"Reachable fallthrough past end at bytecode offset {pc}")
            prior = states.get(target)
            if prior is not None:
                if depth != prior[0]:
                    raise ClassFileError(f"Inconsistent stack depth at bytecode offset {target}")
                merged = prior[1] & initialized
                if merged == prior[1]:
                    continue
            else:
                merged = initialized
            states[target] = (depth, merged)
            if target not in queued:
                queued.add(target)
                pending.append(target)


def load_method(path: str | Path, name: str, descriptor: str) -> Method:
    """Load a static ``(I*)I`` method, rejecting unsupported execution features."""
    if not descriptor.startswith("(") or not descriptor.endswith(")I") or any(c != "I" for c in descriptor[1:-2]):
        raise UnsupportedFeature("Only integer parameters and an integer return, (I*)I, are supported")
    parameters = len(descriptor) - 3
    if parameters > 255:
        raise ClassFileError("Method descriptor exceeds 255 parameter slots")
    reader = _Reader(Path(path).read_bytes())
    if reader.u4() != 0xCAFEBABE:
        raise ClassFileError("Invalid class-file magic")
    minor, major = reader.u2(), reader.u2()
    if not 45 <= major <= 65 or (minor != 0 and not (major == 45 and minor <= 3)):
        raise UnsupportedFeature(f"Unsupported class-file version {major}.{minor}")
    pool = _Pool(reader, major)
    flags, this_class, super_class = reader.u2(), reader.u2(), reader.u2()
    pool.text(pool.get(this_class, 7))
    if flags & (0x0200 | 0x8000):
        raise UnsupportedFeature("Interfaces and module descriptors are not supported")
    if super_class == 0 or pool.text(pool.get(super_class, 7)) != "java/lang/Object":
        raise UnsupportedFeature("Only classes directly extending java/lang/Object are supported")
    interfaces = reader.u2()
    for _ in range(interfaces):
        pool.get(reader.u2(), 7)
    if interfaces:
        raise UnsupportedFeature("Implemented interfaces are not supported")
    fields = set()
    for _ in range(reader.u2()):
        reader.u2()
        field = (pool.text(reader.u2()), pool.text(reader.u2()))
        if field in fields:
            raise ClassFileError("Duplicate field name and descriptor")
        fields.add(field)
        for _name, _attribute in _attributes(reader, pool):
            pass
    selected = None
    methods = set()
    for _ in range(reader.u2()):
        method_flags = reader.u2()
        method_name, method_descriptor = pool.text(reader.u2()), pool.text(reader.u2())
        signature = (method_name, method_descriptor)
        if signature in methods:
            raise ClassFileError("Duplicate method name and descriptor")
        methods.add(signature)
        if method_name == "<clinit>":
            raise UnsupportedFeature("Class initialization (<clinit>) is not supported")
        body = None
        for attribute_name, attribute in _attributes(reader, pool):
            if attribute_name == "Code":
                if body is not None:
                    raise ClassFileError("Duplicate Code attribute")
                body = _code_attribute(attribute, pool)
        if method_flags & (0x0100 | 0x0400):
            if body is not None:
                raise ClassFileError("Native or abstract method has a Code attribute")
        elif body is None:
            raise ClassFileError("Concrete method is missing its Code attribute")
        if signature == (name, descriptor):
            if not method_flags & 0x0008 or method_flags & (0x0100 | 0x0400 | 0x0020):
                raise UnsupportedFeature("Selected method must be static, non-native, concrete, and unsynchronized")
            if name.startswith("<"):
                raise UnsupportedFeature("Special JVM methods cannot be selected")
            code, max_stack, max_locals, handlers = body
            if handlers:
                raise UnsupportedFeature("Exception handlers are not supported")
            selected = Method(name, descriptor, code, max_stack, max_locals, parameters, tuple(pool.constants))
    for _name, _attribute in _attributes(reader, pool):
        pass
    reader.finish()
    if selected is None:
        raise ClassFileError(f"Method {name}{descriptor} not found")
    _verify(selected.code, selected.max_stack, selected.max_locals, parameters, pool)
    return selected
