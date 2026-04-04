# 日志配置说明

## 概述

本项目使用 `tracing` 和 `tracing-subscriber` 实现统一的日志系统，支持本地开发和发布版本两种模式。

## 日志目录

### 本地开发/测试模式
- **日志目录**: 当前工作目录下的 `logs/` 文件夹
- **输出方式**: 同时输出到控制台和日志文件
- **日志格式**: 详细格式（包含文件名、行号、目标模块）

### Release 版本 - macOS App Bundle
- **日志目录**: `~/Library/Logs/aphelios/`
- **输出方式**: 同时输出到控制台和日志文件
- **日志文件命名**: `aphelios_YYYYMMDD_HHMMSS.log`

### Release 版本 - CLI 工具
- **日志目录**: 可执行文件同级目录下的 `logs/` 文件夹
- **输出方式**: 同时输出到控制台和日志文件
- **日志文件命名**: `aphelios_YYYYMMDD_HHMMSS.log`

## 日志级别控制

### 通过环境变量设置

```bash
# 显示所有日志
export RUST_LOG=info

# 只显示特定模块的日志
export RUST_LOG=aphelios_asr=debug,aphelios_core=info

# 关闭某些模块的日志
export RUST_LOG=info,ort=off,h2=off,hyper=off

# 显示调试级别日志
export RUST_LOG=debug
```

### 默认配置

默认的日志过滤器配置为：
```
info,ort=off,h2=off,hyper=off
```

这会：
- 显示 `info` 及以上级别的日志
- 关闭 `ort`（ONNX Runtime）的日志
- 关闭 `h2`（HTTP/2）的日志
- 关闭 `hyper`（HTTP 库）的日志

## 使用示例

### 在代码中初始化日志

```rust
use aphelios_core::init_logging;

fn main() {
    // 初始化日志系统
    init_logging();
    
    // 使用 tracing 宏记录日志
    tracing::info!("Application started");
    tracing::debug!("Loading model from: {}", model_path);
    tracing::warn!("Low memory warning");
    tracing::error!("Failed to process file: {}", error);
    
    // 你的应用逻辑
    // ...
}
```

### 在测试中使用

```rust
#[tokio::test]
async fn my_test() -> Result<()> {
    init_logging();
    tracing::info!("Starting test");
    
    // 测试逻辑
    
    tracing::info!("Test completed successfully");
    Ok(())
}
```

## 日志文件管理

### 查看日志文件

```bash
# macOS App Bundle 日志
ls -lht ~/Library/Logs/aphelios/

# CLI 工具日志（假设安装在 release_bin）
ls -lht release_bin/logs/

# 当前目录日志（开发时）
ls -lht logs/
```

### 清理旧日志

日志文件按时间戳命名，可以安全删除旧文件：

```bash
# 清理 7 天前的日志
find ~/Library/Logs/aphelios/ -name "*.log" -mtime +7 -delete
```

## 代码实现

### 核心函数

日志系统实现在 `aphelios_core/src/utils/logger.rs`：

- `init_logging()`: 主日志初始化函数
- `init_test_logging()`: 测试用的简化日志初始化
- `get_log_dir()`: 确定日志目录路径

### 日志层级

日志系统使用双层架构：
1. **控制台层**: 使用 `pretty` 格式，方便开发时查看
2. **文件层**: 使用 `compact` 格式，节省磁盘空间

## 依赖项

相关依赖在 `Cargo.toml` 中配置：

```toml
tracing = "0.1.44"
tracing-subscriber = "0.3.18"
tracing-appender = "0.2"
chrono = "0.4"
dirs = "6.0.0"
```

## 注意事项

1. **单次初始化**: `init_logging()` 使用 `Once` 确保只初始化一次
2. **容错处理**: 如果日志文件创建失败，应用仍会正常运行
3. **线程安全**: 日志系统完全线程安全，可在多线程环境下使用
4. **性能**: Release 版本已优化，日志开销最小化

## 故障排查

### 日志文件未创建

检查：
1. 是否调用了 `init_logging()`
2. 是否有写入权限到日志目录
3. 查看控制台输出是否有初始化错误信息

### 日志过多或过少

调整 `RUST_LOG` 环境变量：
```bash
# 减少日志
export RUST_LOG=warn

# 增加调试信息
export RUST_LOG=aphelios_asr=debug,aphelios_ocr=debug
```

### macOS App 无法写入日志

检查 `~/Library/Logs/aphelios/` 目录权限：
```bash
ls -la ~/Library/Logs/ | grep aphelios
chmod 755 ~/Library/Logs/aphelios
```
