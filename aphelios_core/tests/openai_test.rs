use anyhow::Result;
use aphelios_core::{
    openai::{translate_file_zh_hant_zh_hans, translate_zh_hant_zh_hans},
    utils::core_utils,
};
use tracing::info;

#[tokio::test]
async fn translategemma_file() -> Result<()> {
    core_utils::init_tracing();
    let txt_path = "/Users/larry/Documents/docs/zhongyixiyi.txt";

    let _ = translate_file_zh_hant_zh_hans(txt_path).await;
    Ok(())
}

#[tokio::test]
async fn translategemma_test() -> Result<()> {
    core_utils::init_tracing();
    let msg = "半個世紀後，廣受敬重的中國公衛先驅陳志潛（一九○三—二○○○）這樣回顧這場歷史性的衝突";
    let result = translate_zh_hant_zh_hans(vec![msg]).await?;
    for r in result {
        info!("result : {}", r);
    }
    Ok(())
}
