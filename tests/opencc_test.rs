use anyhow::{Error as E, Result};
use aphelios_cli::opencc::utils;
use once_cell::sync::Lazy;
use tracing::{Level, info};

static TRACING: Lazy<()> = Lazy::new(|| {
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(Level::TRACE)
        .finish();
    let _ = tracing::subscriber::set_global_default(subscriber);
});

#[test]
fn opencc_test() -> Result<()> {
    Lazy::force(&TRACING);

    let text = "哈維爾的診斷在羅馬尼亞幾乎無人理解。該國結合了各體制最糟的一面：奉行新史達林主義的東德安全國家、失能的波蘭經濟，還有拉丁美洲各國的獨裁統治，再將其糟糕程度拓展到新境界。羅馬尼亞社會主義有位領頭學生稱之為民族史達林主義。比起其他共產集團國家，恐懼更是深入民眾的日常生活。羅馬尼亞人深知，只要說了什麼政治不正確的話，就可能會消失在監獄中，然後徹底從世上消失。羅馬尼亞共產黨政權下有五十萬政治犯，估計就有十萬人死於獄中。[85]";
    let r = utils::t2s(text);
    // 史達林 史达林
    info!(" t2s result {}", r);
    Ok(())
}
