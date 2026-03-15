#!/bin/bash

# 设置错误即退出
set -e

# 获取脚本所在目录（项目根目录）
PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$PROJECT_ROOT"

echo "🚀 开始构建瘦身版本..."

# 1. 清理旧构建（可选）
# cargo clean

# 2. 构建 Slint UI 和 CLI 工具
echo "📦 正在编译二进制文件 (Metal 加速)..."
cargo build --release -p aphelios_slint -p aphelios_tool --features metal

# 3. 准备输出目录
OUTPUT_DIR="$PROJECT_ROOT/release_bin"
mkdir -p "$OUTPUT_DIR"

# 4. 复制二进制文件
cp "target/release/aphelios_slint" "$OUTPUT_DIR/"
cp "target/release/aphelios_tool" "$OUTPUT_DIR/"

# 5. 使用 UPX 进一步压缩（如果系统已安装 upx）
if command -v upx >/dev/null 2>&1; then
    echo "💎 正在使用 UPX 压缩二进制文件..."
    # --best 压缩率最高，但耗时稍长
    upx --best "$OUTPUT_DIR/aphelios_slint" || true
    upx --best "$OUTPUT_DIR/aphelios_tool" || true
else
    echo "⚠️ 未安装 UPX，跳过二进制压缩步骤。建议安装 UPX 以获得更小体积。"
fi

echo "✅ 构建完成！"
echo "📍 文件位置: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
