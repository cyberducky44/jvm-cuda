// Compiled directly by NVRTC: no host runtime or project headers are required.
// The class-file loader checks instruction boundaries, constant-pool types,
// initialized locals, and stack-depth consistency before this kernel is called.
// This kernel additionally bounds every code/frame access. The ABI deliberately
// has no constant-pool length, so ldc indexes remain a host-validated precondition.

static __device__ __forceinline__ bool available(int size, int offset, int count) {
    return offset >= 0 && offset <= size && count >= 0 && count <= size - offset;
}

static __device__ __forceinline__ unsigned int read_u16(const unsigned char* p) {
    return (static_cast<unsigned int>(p[0]) << 8) | p[1];
}

static __device__ __forceinline__ unsigned int read_u32(const unsigned char* p) {
    return (static_cast<unsigned int>(p[0]) << 24)
        | (static_cast<unsigned int>(p[1]) << 16)
        | (static_cast<unsigned int>(p[2]) << 8) | p[3];
}

// Avoid implementation-defined unsigned-to-signed conversions and signed
// overflow, including when the bit pattern denotes Integer.MIN_VALUE.
static __device__ __forceinline__ int signed_int(unsigned int value) {
    return (value & 0x80000000u) ? -1 - static_cast<int>(~value)
                                 : static_cast<int>(value);
}

static __device__ __forceinline__ int signed_short(unsigned int value) {
    return static_cast<int>(value) - ((value & 0x8000u) ? 65536 : 0);
}

static __device__ __forceinline__ int signed_byte(unsigned int value) {
    return static_cast<int>(value) - ((value & 0x80u) ? 256 : 0);
}

static __device__ __forceinline__ bool branch_target(
    int origin, int displacement, int code_size, int& target
) {
    const long long destination = static_cast<long long>(origin) + displacement;
    if (destination < 0 || destination >= code_size) {
        return false;
    }
    target = static_cast<int>(destination);
    return true;
}

extern "C" __global__ void interpret(
    const unsigned char* code, int code_size,
    const int* constants,
    const int* arguments, int argument_count,
    unsigned int* frames, int max_locals, int max_stack,
    int* results, int* statuses, int* error_pcs,
    unsigned long long step_limit, int invocation_count
) {
    const unsigned long long invocation =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (invocation_count <= 0 || invocation >= static_cast<unsigned int>(invocation_count)) {
        return;
    }
    // Missing output buffers cannot carry a diagnostic; never dereference them.
    if (!results || !statuses || !error_pcs) {
        return;
    }
    results[invocation] = 0;
    statuses[invocation] = 3;
    error_pcs[invocation] = 0;
    if (!code || code_size <= 0 || max_locals < 0 || max_stack < 0
        || argument_count < 0 || argument_count > max_locals || step_limit == 0
        || (argument_count != 0 && !arguments)
        || ((max_locals != 0 || max_stack != 0) && !frames)) {
        return;
    }

    const unsigned long long stride =
        static_cast<unsigned long long>(max_locals) + max_stack;
    unsigned int* const locals = stride ? frames + invocation * stride : frames;
    unsigned int* const stack = max_locals ? locals + max_locals : locals;
    const unsigned long long argument_base = invocation * argument_count;
    for (int index = 0; index < argument_count; ++index) {
        locals[index] = static_cast<unsigned int>(arguments[argument_base + index]);
    }

    int pc = 0;
    int depth = 0;
    int status = 3;
    unsigned long long steps = 0;

#define REQUIRE(condition) do { if (!(condition)) goto finished; } while (false)

    for (;;) {
        REQUIRE(available(code_size, pc, 1));
        if (steps == step_limit) {
            status = 2;
            goto finished;
        }
        ++steps;
        const unsigned int opcode = code[pc];
        int next_pc = pc + 1;
        switch (opcode) {
            case 0x00: // nop
                break;
            case 0x02: case 0x03: case 0x04: case 0x05:
            case 0x06: case 0x07: case 0x08: // iconst_m1 .. iconst_5
                REQUIRE(depth < max_stack);
                stack[depth++] = opcode - 3u;
                break;
            case 0x10: // bipush
                REQUIRE(available(code_size, next_pc, 1) && depth < max_stack);
                stack[depth++] = static_cast<unsigned int>(signed_byte(code[next_pc]));
                ++next_pc;
                break;
            case 0x11: // sipush
                REQUIRE(available(code_size, next_pc, 2) && depth < max_stack);
                stack[depth++] = static_cast<unsigned int>(signed_short(read_u16(code + next_pc)));
                next_pc += 2;
                break;
            case 0x12: case 0x13: { // ldc / ldc_w (integer constants only)
                const int width = opcode == 0x12 ? 1 : 2;
                REQUIRE(available(code_size, next_pc, width) && depth < max_stack);
                const unsigned int index = width == 1 ? code[next_pc] : read_u16(code + next_pc);
                REQUIRE(constants && index != 0);
                stack[depth++] = static_cast<unsigned int>(constants[index]);
                next_pc += width;
                break;
            }
            case 0x15: case 0x1a: case 0x1b: case 0x1c: case 0x1d: { // iload[_n]
                unsigned int index;
                if (opcode == 0x15) {
                    REQUIRE(available(code_size, next_pc, 1));
                    index = code[next_pc++];
                } else {
                    index = opcode - 0x1a;
                }
                REQUIRE(index < static_cast<unsigned int>(max_locals) && depth < max_stack);
                stack[depth++] = locals[index];
                break;
            }
            case 0x36: case 0x3b: case 0x3c: case 0x3d: case 0x3e: { // istore[_n]
                unsigned int index;
                if (opcode == 0x36) {
                    REQUIRE(available(code_size, next_pc, 1));
                    index = code[next_pc++];
                } else {
                    index = opcode - 0x3b;
                }
                REQUIRE(index < static_cast<unsigned int>(max_locals) && depth >= 1);
                locals[index] = stack[--depth];
                break;
            }
            case 0x57: // pop
                REQUIRE(depth >= 1);
                --depth;
                break;
            case 0x59: // dup
                REQUIRE(depth >= 1 && depth < max_stack);
                stack[depth] = stack[depth - 1];
                ++depth;
                break;
            case 0x5f: { // swap
                REQUIRE(depth >= 2);
                const unsigned int top = stack[depth - 1];
                stack[depth - 1] = stack[depth - 2];
                stack[depth - 2] = top;
                break;
            }
            case 0x60: case 0x64: case 0x68: // iadd / isub / imul
            case 0x6c: case 0x70: // idiv / irem
            case 0x78: case 0x7a: case 0x7c: // ishl / ishr / iushr
            case 0x7e: case 0x80: case 0x82: { // iand / ior / ixor
                REQUIRE(depth >= 2);
                const unsigned int right = stack[--depth];
                const unsigned int left = stack[depth - 1];
                unsigned int value;
                switch (opcode) {
                    case 0x60: value = left + right; break;
                    case 0x64: value = left - right; break;
                    case 0x68: value = left * right; break;
                    case 0x6c: case 0x70:
                        if (right == 0) {
                            status = 1;
                            goto finished;
                        }
                        if (left == 0x80000000u && right == 0xffffffffu) {
                            value = opcode == 0x6c ? 0x80000000u : 0u;
                        } else if (opcode == 0x6c) {
                            value = static_cast<unsigned int>(signed_int(left) / signed_int(right));
                        } else {
                            value = static_cast<unsigned int>(signed_int(left) % signed_int(right));
                        }
                        break;
                    case 0x78: value = left << (right & 31u); break;
                    case 0x7a:
                        value = (left & 0x80000000u)
                            ? ~(~left >> (right & 31u)) : left >> (right & 31u);
                        break;
                    case 0x7c: value = left >> (right & 31u); break;
                    case 0x7e: value = left & right; break;
                    case 0x80: value = left | right; break;
                    default: value = left ^ right; break; // 0x82 only
                }
                stack[depth - 1] = value;
                break;
            }
            case 0x74: // ineg
                REQUIRE(depth >= 1);
                stack[depth - 1] = 0u - stack[depth - 1];
                break;
            case 0x84: { // iinc
                REQUIRE(available(code_size, next_pc, 2));
                const unsigned int index = code[next_pc];
                REQUIRE(index < static_cast<unsigned int>(max_locals));
                locals[index] += static_cast<unsigned int>(signed_byte(code[next_pc + 1]));
                next_pc += 2;
                break;
            }
            case 0x99: case 0x9a: case 0x9b: case 0x9c: case 0x9d: case 0x9e:
            case 0x9f: case 0xa0: case 0xa1: case 0xa2: case 0xa3: case 0xa4: {
                REQUIRE(available(code_size, next_pc, 2));
                const bool pair = opcode >= 0x9f;
                REQUIRE(depth >= (pair ? 2 : 1));
                int target;
                REQUIRE(branch_target(pc, signed_short(read_u16(code + next_pc)), code_size, target));
                next_pc += 2;
                const int right = pair ? signed_int(stack[--depth]) : 0;
                const int left = signed_int(stack[--depth]);
                const unsigned int comparison = opcode - (pair ? 0x9fu : 0x99u);
                bool taken;
                switch (comparison) {
                    case 0: taken = left == right; break;
                    case 1: taken = left != right; break;
                    case 2: taken = left < right; break;
                    case 3: taken = left >= right; break;
                    case 4: taken = left > right; break;
                    default: taken = left <= right; break;
                }
                if (taken) {
                    next_pc = target;
                }
                break;
            }
            case 0xa7: case 0xc8: { // goto / goto_w
                const int width = opcode == 0xa7 ? 2 : 4;
                REQUIRE(available(code_size, next_pc, width));
                const int displacement = width == 2
                    ? signed_short(read_u16(code + next_pc)) : signed_int(read_u32(code + next_pc));
                REQUIRE(branch_target(pc, displacement, code_size, next_pc));
                break;
            }
            case 0xaa: case 0xab: { // tableswitch / lookupswitch
                REQUIRE(depth >= 1);
                const int padding = (4 - (next_pc & 3)) & 3;
                REQUIRE(available(code_size, next_pc, padding));
                next_pc += padding;
                const int header_size = opcode == 0xaa ? 12 : 8;
                REQUIRE(available(code_size, next_pc, header_size));
                int displacement = signed_int(read_u32(code + next_pc));
                const int key = signed_int(stack[--depth]);
                if (opcode == 0xaa) {
                    const int low = signed_int(read_u32(code + next_pc + 4));
                    const int high = signed_int(read_u32(code + next_pc + 8));
                    REQUIRE(low <= high);
                    const long long count = static_cast<long long>(high) - low + 1;
                    const int entries = next_pc + 12;
                    REQUIRE(count <= (code_size - entries) / 4);
                    if (key >= low && key <= high) {
                        const int index = static_cast<int>(static_cast<long long>(key) - low);
                        displacement = signed_int(read_u32(code + entries + index * 4));
                    }
                } else {
                    const int count = signed_int(read_u32(code + next_pc + 4));
                    const int entries = next_pc + 8;
                    REQUIRE(count >= 0 && count <= (code_size - entries) / 8);
                    // The loader validates strictly increasing signed keys.
                    int first = 0;
                    int last = count;
                    while (first < last) {
                        const int middle = first + (last - first) / 2;
                        const int offset = entries + middle * 8;
                        const int candidate = signed_int(read_u32(code + offset));
                        if (key < candidate) {
                            last = middle;
                        } else if (key > candidate) {
                            first = middle + 1;
                        } else {
                            displacement = signed_int(read_u32(code + offset + 4));
                            break;
                        }
                    }
                }
                REQUIRE(branch_target(pc, displacement, code_size, next_pc));
                break;
            }
            case 0xac: // ireturn discards the rest of the current frame's stack.
                REQUIRE(depth >= 1);
                results[invocation] = signed_int(stack[depth - 1]);
                status = 0;
                goto finished;
            case 0xc4: { // wide iload / istore / iinc only
                REQUIRE(available(code_size, next_pc, 3));
                const unsigned int modified = code[next_pc];
                const unsigned int index = read_u16(code + next_pc + 1);
                REQUIRE(index < static_cast<unsigned int>(max_locals));
                next_pc += 3;
                switch (modified) {
                    case 0x15:
                        REQUIRE(depth < max_stack);
                        stack[depth++] = locals[index];
                        break;
                    case 0x36:
                        REQUIRE(depth >= 1);
                        locals[index] = stack[--depth];
                        break;
                    case 0x84:
                        REQUIRE(available(code_size, next_pc, 2));
                        locals[index] += static_cast<unsigned int>(signed_short(read_u16(code + next_pc)));
                        next_pc += 2;
                        break;
                    default:
                        goto finished;
                }
                break;
            }
            default:
                goto finished;
        }
        // Falling off the method is malformed; retain the faulting opcode PC.
        REQUIRE(next_pc < code_size);
        pc = next_pc;
    }

finished:
    statuses[invocation] = status;
    error_pcs[invocation] = status == 0 ? -1 : pc;
#undef REQUIRE
}
