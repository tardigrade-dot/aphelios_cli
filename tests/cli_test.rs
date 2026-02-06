use aphelios_cli::commands::qwen_llm::qwen_infer;
use assert_cmd::cargo::cargo_bin_cmd;
use anyhow::Result;
use std::time::Instant;
use tracing::{debug, error, info, span, warn, Level};
use tracing_subscriber;

#[test]
fn help_works() {
    let mut cmd = cargo_bin_cmd!("aphelios_cli");
    cmd.arg("--help")
        .assert()
        .success();
}

#[test]
fn test_onnx_infer(){

    let subscriber = tracing_subscriber::fmt()
        // filter spans/events with level TRACE or higher.
        .with_max_level(Level::TRACE)
        // build but do not install the subscriber.
        .finish();
    println!("start test onnx infer...");

    tracing::subscriber::with_default(subscriber, || {

        let start = Instant::now();
        if let Err(e) = llm_infer("假设我们现在在古代,如何证明地球是圆的,给出5种方法") {
            println!("ONNX inference test failed: {}", e);
        } else {
            println!("ONNX inference test completed successfully");
        }
        let duration = start.elapsed();
        info!("Time elapsed: {:?}", duration);
    });
}

pub fn llm_infer(prompt: &str) -> Result<String, String> {
    let chat_prompt = format!("{}<|im_end|>", prompt);
    
    let path_str = "/Users/larry/test_dir/Qwen3-0.6B";
    let model_path = path_str;
    
    let reply = qwen_infer(
        model_path,
        &chat_prompt,
        false,
        None, None, None, None, None, None
    )
    .map_err(|e| e.to_string());

    reply
}
