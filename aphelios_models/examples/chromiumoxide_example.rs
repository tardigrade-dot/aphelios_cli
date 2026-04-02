use aphelios_core::init_logging;
use chromiumoxide::Browser;
use futures::StreamExt;
use tokio::time::{sleep, Duration};
use tracing::info;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging();
    let (browser, mut handler) = Browser::connect("http://localhost:9222").await?;

    tokio::spawn(async move { while let Some(_) = handler.next().await {} });

    let page = browser.new_page("https://www.bilibili.com").await?;
    println!(
        "页面已打开: {}",
        page.get_title().await?.unwrap_or_default()
    );

    sleep(Duration::from_secs(2)).await;

    // 先点击聚焦
    page.find_element(".nav-search-input")
        .await?
        .click()
        .await?;

    sleep(Duration::from_millis(300)).await;

    // 用 JS 设置值并触发 Vue 能感知的 input 事件
    page.evaluate(
        r#"
        const el = document.querySelector('.nav-search-input');
        const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
            window.HTMLInputElement.prototype, 'value'
        ).set;
        nativeInputValueSetter.call(el, 'ai教程');
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
    "#,
    )
    .await?;

    sleep(Duration::from_millis(500)).await;

    page.find_element(".nav-search-btn").await?.click().await?;

    sleep(Duration::from_secs(2)).await;
    page.wait_for_navigation().await?;

    let pages = browser.pages().await?;
    let mut search_page = None;
    for p in &pages {
        info!("opener_id : {}", p.opener_id().as_ref().unwrap().inner());
        if let Ok(Some(url)) = p.url().await {
            if url.contains("search.bilibili.com") {
                search_page = Some(p);
                break;
            }
        }
    }
    let search_page = search_page.expect("找不到搜索结果 tab");

    let cards = search_page
        .find_elements(".bili-video-card__info--tit")
        .await?;
    println!("\n搜索结果：");
    for (i, card) in cards.iter().take(5).enumerate() {
        if let Some(title) = card.inner_text().await? {
            println!("{}. {}", i + 1, title.trim());
        }
    }

    Ok(())
}
