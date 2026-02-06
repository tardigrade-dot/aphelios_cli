
use std::time::Duration;
use std::thread::sleep;
use aphelios_cli::{commands::{qwen_llm, qwenvl::{self}}, measure_time};
use anyhow::Result;
use tracing::{Level, info};

#[tokio::test]
async fn qwen_vlm() -> Result<()> {

    let subscriber = tracing_subscriber::fmt()
        .with_max_level(Level::TRACE)
        .finish();
    let _ = tracing::subscriber::set_default(subscriber);
    let res = qwenvl::run_vlm("/Volumes/sw/pretrained_models/Qwen3-VL-2B-Instruct").await;
    res
}

#[tokio::test]
async fn qwen_llm() -> Result<()> {

    let subscriber = tracing_subscriber::fmt()
        .with_max_level(Level::TRACE)
        .finish();
    let _ = tracing::subscriber::set_default(subscriber);
    let res = measure_time!{

        let res = qwenvl::run_llm("/Volumes/sw/pretrained_models/Qwen3-0.6B").await;
        res
    };
    res
}

#[test]
fn measure_time_test() -> Result<()>{

    let subscriber = tracing_subscriber::fmt()
        .with_max_level(Level::TRACE)
        .finish();
    let _ = tracing::subscriber::set_default(subscriber);
    let _ = measure_time!("测试", sleep(Duration::from_secs(10)));
    
    Ok(())
}

// #[tokio::test]
// async fn qwen_book_search() -> Result<()>{

//     let subscriber = tracing_subscriber::fmt()
//         .with_max_level(Level::TRACE)
//         .finish();
//     let _ = tracing::subscriber::set_default(subscriber);

//     let res = qwenvl::book_search("/Volumes/sw/pretrained_models/gemma-3-1b-it").await;
//     info!("{}", res?);
//     Ok(())
// }

// #[test]
// fn candle_qwen_test() -> Result<()>{

//     let res = qwen_llm::qwen_infer("/Volumes/sw/pretrained_models/Qwen3-0.6B", 
//     "介绍一下苏格拉底", false, None, None, None, None, None, None)?;
//     print!("{}", res);
//     Ok(())
// }