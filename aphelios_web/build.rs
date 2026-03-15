use std::process::Command;
use std::path::PathBuf;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let src_web_dir = PathBuf::from(&manifest_dir).join("src-web");
    let web_dist_dir = PathBuf::from(&manifest_dir).join("web-dist");

    // 告诉 cargo 如果 src-web 目录变化则重新运行
    println!("cargo:rerun-if-changed=src-web");
    println!("cargo:rerun-if-changed={}", src_web_dir.display());

    // 设置 rust-embed 的资源目录环境变量
    println!("cargo:rustc-env=RUST_EMBED_SEARCH_PATH={}", web_dist_dir.display());

    // 构建前端项目
    Command::new("npm")
        .args(["install"])
        .current_dir(&src_web_dir)
        .status()
        .expect("Failed to run npm install");

    Command::new("npm")
        .args(["run", "build"])
        .current_dir(&src_web_dir)
        .status()
        .expect("Failed to run npm run build");
}
