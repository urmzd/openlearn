//! quant_dot.zig — pure-Zig INT8 symmetric quantization + SIMD dot product.
//!
//! The CPU twin of ../../quantization/code/symmetric_quant.cu: same math
//! (scale = absmax/127, q = clamp(round(x/s), -127, 127), out = s_a*s_b*acc),
//! but the matmul inner loop uses Zig's portable `@Vector` SIMD instead of
//! CUDA's dp4a. Shows the "pure Zig kernel" idiom from the frameworks README §5.
//!
//! No external dependencies. Targets Zig 0.13/0.14 (the std API and `@Vector`
//! builtins move between releases — check `zig version`).
//!
//! Build & run:  zig run quant_dot.zig -O ReleaseFast

const std = @import("std");

const VLEN = 16; // i8 lanes per SIMD op; products (<=127*127) fit in i32

/// scale = max(|x|) / 127  (per-tensor symmetric)
fn absmaxScale(x: []const f32) f32 {
    var m: f32 = 0;
    for (x) |v| m = @max(m, @abs(v));
    return if (m > 0) m / 127.0 else 1.0;
}

/// q = clamp(round(x / scale), -127, 127)
fn quantize(x: []const f32, scale: f32, out: []i8) void {
    const inv = 1.0 / scale;
    for (x, 0..) |v, i| {
        const q = std.math.clamp(@round(v * inv), -127.0, 127.0);
        out[i] = @intFromFloat(q);
    }
}

/// SIMD INT8 dot product: widen i8 lanes to i32, multiply, horizontal-add.
fn dotI8(a: []const i8, b: []const i8) i32 {
    std.debug.assert(a.len == b.len);
    var acc: i32 = 0;
    var i: usize = 0;
    while (i + VLEN <= a.len) : (i += VLEN) {
        const va: @Vector(VLEN, i8) = a[i..][0..VLEN].*;
        const vb: @Vector(VLEN, i8) = b[i..][0..VLEN].*;
        const wa: @Vector(VLEN, i32) = va; // element-wise i8 -> i32 widening
        const wb: @Vector(VLEN, i32) = vb;
        acc += @reduce(.Add, wa * wb); // horizontal sum of the lane products
    }
    while (i < a.len) : (i += 1) acc += @as(i32, a[i]) * @as(i32, b[i]); // tail
    return acc;
}

pub fn main() !void {
    const n = 4096;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const fa = try alloc.alloc(f32, n);
    const fb = try alloc.alloc(f32, n);
    const qa = try alloc.alloc(i8, n);
    const qb = try alloc.alloc(i8, n);
    defer alloc.free(fa);
    defer alloc.free(fb);
    defer alloc.free(qa);
    defer alloc.free(qb);

    var ref: f64 = 0;
    for (0..n) |i| {
        const t: f32 = @floatFromInt(i);
        fa[i] = @sin(0.01 * t) * 3.0;
        fb[i] = @cos(0.02 * t);
    }
    fa[7] = 25.0; // outlier inflates the per-tensor scale (frameworks/quant lesson)
    for (0..n) |i| ref += @as(f64, fa[i]) * @as(f64, fb[i]);

    const s_a = absmaxScale(fa);
    const s_b = absmaxScale(fb);
    quantize(fa, s_a, qa);
    quantize(fb, s_b, qb);

    const acc = dotI8(qa, qb);
    const got: f64 = @as(f64, s_a) * @as(f64, s_b) * @as(f64, @floatFromInt(acc));

    const rel = @abs(got - ref) / @abs(ref) * 100.0;
    std.debug.print("scale_a={d:.5}  scale_b={d:.5}\n", .{ s_a, s_b });
    std.debug.print("fp32 dot = {d:.4}\n", .{ref});
    std.debug.print("int8 dot = {d:.4}   rel.err = {d:.4}%\n", .{ got, rel });
    std.debug.print("(the outlier at fa[7] widens scale_a -> bigger step -> more error;\n" ++
        " per-channel/per-group scales would fix it — same lesson as the CUDA twin.)\n", .{});
}
