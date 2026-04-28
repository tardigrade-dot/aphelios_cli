运行一个run指令: cargo run -- run --verbose

cargo clean
创建: cargo new hello_cli
安装指令: cargo install --path .
卸载指令: cargo uninstall aphelios_cli
运行指令: aphelios_cli run --verbose

cargo run --example texify2_onnx

aphelios_cli init /a/b

cargo run -p aphelios_tool -- run --verbose

sudo cargo flamegraph --release -p aphelios_core --test tts_test -- qwen3_test --nocapture

cargo run --release --example qwen3_tts_streaming --features metal

cargo run --release --example qwen3_tts_streaming_cv --features metal

cargo run --package aphelios_device --release --example realtime_camera_example --features metal

sudo cargo flamegraph --package aphelios_device --example realtime_camera_example --features metal

sudo cargo flamegraph --release --example qwen3_tts_streaming --features metal -- --nocapture

cargo test --release -p aphelios_asr --test whisper_asr_test --features metal -- asr_test_16k

cargo test --release -p aphelios_asr --test whisper_rs_test --features metal -- coreml_test

cargo test --release -p aphelios_core --test rtdetr_example --features metal -- label_video_test --exact --nocapture

cargo tree > tree.log

 ./build_release.sh --features metal

# 默认端口 3000
cargo run -p aphelios_web

# 指定端口
cargo run -p aphelios_web -- 8080

# 指定地址和端口
cargo run -p aphelios_web -- 127.0.0.1:8080

whisper
wav short:2min long:30min
ggml 15s 400s
candle 38s 305s

日志: ~/Library/Logs/aphelios/
配置文件: ~/Library/Application\ Support/aphelios_cli
