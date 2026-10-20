//! llama_cpp.zig — calling llama.cpp's C API directly from Zig via `@cImport`.
//!
//! The "wrap C" idiom from the frameworks README §5: Zig reads `llama.h`
//! directly — no bindgen, no FFI crate, no glue. This is Zig's headline feature
//! for inference work: reuse the most-deployed engine (llama.cpp) with C-level
//! control and a tiny binary.
//!
//! ILLUSTRATIVE — not built in CI. It needs a llama.cpp checkout (headers +
//! libllama) and a GGUF model. llama.cpp's API tracks upstream and changes
//! often; the symbol names below match the late-2025 API (model/vocab split,
//! sampler chains). Verify against the `llama.h` you link.
//!
//! build.zig (sketch):
//!   const exe = b.addExecutable(.{ .name = "llama_zig", .root_source_file = b.path("llama_cpp.zig"), .target = target, .optimize = optimize });
//!   exe.addIncludePath(b.path("llama.cpp/include"));
//!   exe.addIncludePath(b.path("llama.cpp/ggml/include"));
//!   exe.addLibraryPath(b.path("llama.cpp/build/bin"));
//!   exe.linkSystemLibrary("llama");
//!   exe.linkLibC();
//! Run:  ./llama_zig model.gguf "Once upon a time"

const std = @import("std");

// Zig reads the C header verbatim — this is the whole "binding".
const c = @cImport({
    @cInclude("llama.h");
});

pub fn main() !void {
    var args = std.process.args();
    _ = args.skip();
    const model_path = args.next() orelse "model.gguf";
    const prompt = args.next() orelse "Once upon a time";

    // 1. backend + model
    c.llama_backend_init();
    defer c.llama_backend_free();

    var mparams = c.llama_model_default_params();
    mparams.n_gpu_layers = 99; // offload to GPU if a backend is compiled in
    const model = c.llama_model_load_from_file(model_path.ptr, mparams) orelse {
        std.debug.print("failed to load model: {s}\n", .{model_path});
        return error.ModelLoad;
    };
    defer c.llama_model_free(model);
    const vocab = c.llama_model_get_vocab(model);

    // 2. context (holds the KV cache)
    var cparams = c.llama_context_default_params();
    cparams.n_ctx = 2048;
    const ctx = c.llama_init_from_model(model, cparams) orelse return error.Context;
    defer c.llama_free(ctx);

    // 3. tokenize the prompt
    var tokens: [512]c.llama_token = undefined;
    const n_tokens = c.llama_tokenize(
        vocab,
        prompt.ptr,
        @intCast(prompt.len),
        &tokens,
        tokens.len,
        true, // add BOS
        false, // parse special tokens
    );
    if (n_tokens < 0) return error.Tokenize;

    // 4. greedy sampler chain
    const smpl = c.llama_sampler_chain_init(c.llama_sampler_chain_default_params());
    defer c.llama_sampler_free(smpl);
    c.llama_sampler_chain_add(smpl, c.llama_sampler_init_greedy());

    // 5. decode loop: prefill the prompt, then generate token-by-token
    var batch = c.llama_batch_get_one(&tokens, n_tokens);
    const stdout = std.io.getStdOut().writer();
    var generated: usize = 0;
    while (generated < 128) : (generated += 1) {
        if (c.llama_decode(ctx, batch) != 0) return error.Decode;

        const id = c.llama_sampler_sample(smpl, ctx, -1); // sample from last position
        if (c.llama_vocab_is_eog(vocab, id)) break; // end-of-generation token

        var piece: [256]u8 = undefined;
        const len = c.llama_token_to_piece(vocab, id, &piece, piece.len, 0, false);
        if (len > 0) try stdout.writeAll(piece[0..@intCast(len)]);

        // feed the sampled token back in as the next single-token batch
        var next = [_]c.llama_token{id};
        batch = c.llama_batch_get_one(&next, 1);
    }
    try stdout.writeAll("\n");
}
