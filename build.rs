fn main() {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        println!("cargo:rustc-link-search=native=/opt/homebrew/lib");
        println!("cargo:rustc-link-lib=opencc");
    }

    slint_build::compile("src/ui/app.slint").expect("Slint build failed");
}
