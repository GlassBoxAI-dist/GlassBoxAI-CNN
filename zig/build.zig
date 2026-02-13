// MIT License
//
// Copyright (c) 2025 Matthew Abbott

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const cnn_mod = b.addModule("facaded_cnn_cuda", .{
        .root_source_file = b.path("facaded_cnn_cuda.zig"),
        .target = target,
        .optimize = optimize,
    });
    cnn_mod.addLibraryPath(.{ .cwd_relative = "../target/release" });
    cnn_mod.linkSystemLibrary("facaded_cnn_cuda");

    const lib = b.addStaticLibrary(.{
        .name = "facaded_cnn_cuda_zig",
        .root_source_file = b.path("facaded_cnn_cuda.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib.addLibraryPath(.{ .cwd_relative = "../target/release" });
    lib.linkSystemLibrary("facaded_cnn_cuda");
    b.installArtifact(lib);

    const tests = b.addTest(.{
        .root_source_file = b.path("facaded_cnn_cuda.zig"),
        .target = target,
        .optimize = optimize,
    });
    tests.addLibraryPath(.{ .cwd_relative = "../target/release" });
    tests.linkSystemLibrary("facaded_cnn_cuda");

    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
