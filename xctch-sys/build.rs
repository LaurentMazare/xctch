const ENV_VAR: &str = "EXECUTORCH_DIR";
const ENV_VAR_INCLUDE: &str = "EXECUTORCH_DIR_INCLUDE";
const ENV_VAR_LIB: &str = "EXECUTORCH_DIR_LIB";

fn main() {
    let exec_dir = std::env::var(ENV_VAR).expect(&format!("{ENV_VAR} is not set"));
    let include_dir =
        std::env::var(ENV_VAR_INCLUDE).unwrap_or_else(|_| format!("{exec_dir}/include"));
    let lib_dir = std::env::var(ENV_VAR_LIB).unwrap_or_else(|_| format!("{exec_dir}/lib"));

    cxx_build::bridge("src/cxx_ffi.rs")
        .include(include_dir)
        .std("c++17")
        .file("cpp/cxx_api.cpp")
        .compile("xctch-sys");

    println!("cargo:rerun-if-env-changed={ENV_VAR}");
    println!("cargo:rerun-if-env-changed={ENV_VAR_INCLUDE}");
    println!("cargo:rerun-if-env-changed={ENV_VAR_LIB}");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cpp/cxx_api.hpp");
    println!("cargo:rerun-if-changed=cpp/cxx_api.cpp");
    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rustc-link-lib=static:+whole-archive=executorch");
    println!("cargo:rustc-link-lib=static:+whole-archive=executorch_core");
    println!("cargo:rustc-link-lib=static:+whole-archive=microkernels-prod");
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
