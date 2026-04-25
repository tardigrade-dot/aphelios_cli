use anyhow::Result;
use aphelios_core::openai::infer::{translate_file_zh_hant_zh_hans, translate_zh_hant_zh_hans};
use aphelios_core::utils::logger;
use tracing::info;

#[tokio::test]
async fn translategemma_file() -> Result<()> {
    logger::init_logging();
    let txt_path = "/Users/larry/Documents/docs/zhongyixiyi.txt";

    let _ = translate_file_zh_hant_zh_hans(txt_path).await;
    Ok(())
}

#[tokio::test]
async fn translategemma_test() -> Result<()> {
    logger::init_logging();
    let msg = "藉由探索這個素樸的問題，我發現了一個少為人知的、「非驢非馬」的歷史困局。自民國時期以來，積極推動「中醫科學化」的改革派中醫師幾乎不可避免地陷入這個雙重困局之中。\n如果他們改革的目標只在「保存傳統」，他們只需要負隅頑抗、堅持中醫與現代科學不可共量，甚至超乎科學之上，即可自圓其說。";
    let result = translate_zh_hant_zh_hans(vec![msg]).await?;
    for r in result {
        info!("result : {}", r);
    }
    Ok(())
}
