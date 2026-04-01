#!/bin/bash
set -e
PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$PROJECT_ROOT"

# 接收外部传入的 features，默认为 metal
FEATURES=${1:-"metal"}

echo "🚀 开始构建 (Features: $FEATURES)..."

# 安装 cargo-bundle (如果未安装)
if ! command -v cargo-bundle >/dev/null 2>&1; then
    echo "📦 安装 cargo-bundle..."
    cargo install cargo-bundle
fi

# 编译指定子项目 (包含 bundle 生成)
cargo build --release -p aphelios_slint -p aphelios_tool --features "$FEATURES"

# 使用 cargo-bundle 生成 .app bundle
echo "📦 生成 macOS App Bundle..."
cargo bundle --release -p aphelios_slint --features "$FEATURES"

OUTPUT_DIR="$PROJECT_ROOT/release_bin"
mkdir -p "$OUTPUT_DIR"

# 兼容 Windows 扩展名 (.exe)
EXT=""
[[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]] && EXT=".exe"

# 复制 CLI 工具
cp "target/release/aphelios_tool$EXT" "$OUTPUT_DIR/"

# 复制 .app bundle (cargo-bundle 生成到 target/release/bundle/osx/)
if [[ "$OSTYPE" == darwin* ]]; then
    BUNDLE_NAME="Aphelios.app"
    BUNDLE_PATH="target/release/bundle/osx/$BUNDLE_NAME"
    
    # 从 cargo-bundle 输出目录复制 .app
    if [ -d "$BUNDLE_PATH" ]; then
        cp -r "$BUNDLE_PATH" "$OUTPUT_DIR/"
        echo "✅ App Bundle 已生成：$OUTPUT_DIR/$BUNDLE_NAME"
    else
        echo "⚠️ 警告：未找到 cargo-bundle 生成的 .app"
        echo "   查找路径：$BUNDLE_PATH"
        echo "   请检查："
        echo "   1. aphelios_slint/Cargo.toml 中的 [package.metadata.bundle] 配置"
        echo "   2. cargo bundle 命令是否成功执行"
        ls -la target/release/bundle/ 2>/dev/null || echo "   bundle 目录不存在"
        exit 1
    fi
fi

# UPX 压缩 (仅压缩 CLI 工具，不压缩 .app 内的二进制)
if command -v upx >/dev/null 2>&1; then
    upx --best "$OUTPUT_DIR/aphelios_tool$EXT" || true
fi

echo "✅ 构建完成！输出目录：$OUTPUT_DIR"
