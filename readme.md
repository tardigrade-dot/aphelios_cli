运行一个run指令: cargo run -- run --verbose

cargo clean
创建: cargo new hello_cli
安装指令: cargo install --path .
卸载指令: cargo uninstall aphelios_cli
运行指令: aphelios_cli run --verbose

cargo run --example texify2_onnx

cargo run --profile bench

aphelios_cli qwen-vlm /Volumes/sw/pretrained_models/Qwen3-VL-2B-Instruct

aphelios_cli qwen-llm /Volumes/sw/pretrained_models/Qwen3-0.6B

aphelios_cli init /a/b

sudo cargo flamegraph --release -p aphelios_core --test tts_test -- qwen3_test --nocapture

cargo run --release --example qwen3_tts_streaming --features metal

cargo run --release --example qwen3_tts_streaming_cv --features metal

sudo cargo flamegraph --release --example qwen3_tts_streaming --features metal -- --nocapture

cargo test --release -p aphelios_asr --test whisper_asr_test --features metal -- asr_test_16k

cargo test --release -p aphelios_asr --test whisper_rs_test --features metal -- coreml_test 

whisper
wav short:2min long:30min
ggml 15s 400s 
candle 38s 305s