use aphelios_core::init_logging;
use chromiumoxide::Browser;
use futures::StreamExt;
use tracing::info;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging();
    let (browser, mut handler) = Browser::connect("http://localhost:9222").await?;

    tokio::spawn(async move { while let Some(_) = handler.next().await {} });

    let page = browser.new_page("https://www.bilibili.com").await?;
    info!(
        "页面已打开: {}",
        page.get_title().await?.unwrap_or_default()
    );

    Ok(())
}
