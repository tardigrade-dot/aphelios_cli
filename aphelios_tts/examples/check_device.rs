//! 检查设备配置

use aphelios_core::utils::core_utils;
use candle_core::Device;

fn main() -> anyhow::Result<()> {
    core_utils::init_logging();
    
    println!("🔍 检查设备配置...\n");
    
    // 检查默认设备
    let device = core_utils::get_default_device(false)?;
    println!("默认设备：{:?}", device);
    
    // 检查 Metal 是否可用
    #[cfg(feature = "metal")]
    {
        println!("\n✅ Metal 功能已启用");
        match Device::new_metal(0) {
            Ok(d) => println!("✅ Metal 设备可用：{:?}", d),
            Err(e) => println!("❌ Metal 设备不可用：{}", e),
        }
    }
    
    #[cfg(not(feature = "metal"))]
    {
        println!("\n❌ Metal 功能未启用");
    }
    
    // 检查 CUDA 是否可用
    #[cfg(feature = "cuda")]
    {
        println!("\n✅ CUDA 功能已启用");
        match Device::cuda_if_available(0) {
            Ok(d) => println!("✅ CUDA 设备可用：{:?}", d),
            Err(e) => println!("❌ CUDA 设备不可用：{}", e),
        }
    }
    
    #[cfg(not(feature = "cuda"))]
    {
        println!("\n❌ CUDA 功能未启用");
    }
    
    println!("\n💡 提示：");
    println!("   - 使用 Metal: cargo run --features metal");
    println!("   - 使用 CUDA: cargo run --features cuda");
    println!("   - 仅 CPU: cargo run");
    
    Ok(())
}
