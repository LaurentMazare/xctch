fn main() {
    let exec_dir = std::env::var("EXECUTORCH_DIR").expect("EXECUTORCH_DIR is not set");

    cxx_build::bridge("src/cxx_ffi.rs")
        .include(format!("{exec_dir}/../"))
        .std("c++17")
        .file("cpp/cxx_api.cpp")
        .compile("xctch-sys");

    println!("cargo:rerun-if-env-changed=EXECUTORCH_DIR");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cpp/cxx_api.hpp");
    println!("cargo:rerun-if-changed=cpp/cxx_api.cpp");
    println!("cargo:rustc-link-search=native={exec_dir}/cmake-out/lib");
    println!("cargo:rustc-link-lib=static:+whole-archive=executorch");
    println!("cargo:rustc-link-lib=static:+whole-archive=executorch_no_prim_ops");
    println!("cargo:rustc-link-lib=static:+whole-archive=extension_data_loader");
    println!("cargo:rustc-link-lib=static:+whole-archive=portable_kernels");
    println!("cargo:rustc-link-lib=static:+whole-archive=eigen_blas");
    println!("cargo:rustc-link-lib=static:+whole-archive=optimized_native_cpu_ops_lib");
    println!("cargo:rustc-link-lib=static:+whole-archive=optimized_kernels");

    // XNNPACK
    println!("cargo:rustc-link-lib=static:+whole-archive=xnnpack_backend");
    println!("cargo:rustc-link-lib=static:+whole-archive=XNNPACK");
    println!("cargo:rustc-link-lib=static:+whole-archive=pthreadpool");
    println!("cargo:rustc-link-lib=static:+whole-archive=cpuinfo");
}
