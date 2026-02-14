use anyhow::{Ok, Result};
use aphelios_cli::engine;
use candle_core::{DType, Device};
use fastembed::{EmbeddingModel, InitOptions, Qwen3TextEmbedding, TextEmbedding};
use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
use glob::glob;
use rusqlite::{Connection, params};

#[test]
fn engine_test2() -> Result<()> {
    let mut files: Vec<String> = Vec::new();
    for entry in glob("/Volumes/sw/books/*.epub")?.chain(glob("/Volumes/sw/books/*.pdf")?) {
        let path = entry?;
        if let Some(stem) = path.file_stem() {
            files.push(stem.to_string_lossy().into_owned());
        }
    }

    for i_f in files {
        println!("{}", i_f.replace("(Z-Library)", ""));
    }
    Ok(())
}

#[test]
fn engine_test() -> Result<()> {
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    let model = Qwen3TextEmbedding::from_hf("Qwen/Qwen3-Embedding-0.6B", &device, DType::F32, 512)?;

    const INDEX_FILE_FIELD: &str = "/Users/larry/coderesp/aphelios_cli/output/file_index.usearch";

    let mut se = engine::SearchEngine::new(model, INDEX_FILE_FIELD)?;

    let base_path = "/Volumes/sw/books";

    let mut files: Vec<String> = Vec::new();

    for entry in glob(format!("{}/*.epub", base_path).as_str())?
        .chain(glob(format!("{}/*.pdf", base_path).as_str())?)
    {
        let path = entry?;
        if let Some(stem) = path.file_stem() {
            files.push(stem.to_string_lossy().into_owned());
        }
    }

    let mut clean_files: Vec<String> = Vec::new();
    for i_f in files {
        clean_files.push(i_f.replace("(Z-Library)", ""));
    }

    let _ = se.build_index(clean_files);

    let r = se.search("与东欧历史相关的书籍", 20);

    print!("result----------------------------------");
    for (i, j) in r.unwrap() {
        println!("{}", i);
    }
    print!("result----------------------------------");
    Ok(())
}

#[test]
fn rerank_test() -> Result<()> {
    // With custom options
    let mut model = TextRerank::try_new(
        RerankInitOptions::new(RerankerModel::BGERerankerBase).with_show_download_progress(true),
    )?;
    let txt = "東歐百年史 · 冊1】共同體的神話：東歐的民族主義與社會革命的崛起 (約翰 · 康納利 (John Connelly) 著；羅亞琪 譯) 
【東歐百年史 · 冊2】共同體的神話：極權暴政的席捲與野蠻歐陸的誕生 = From Peoples into Nations A History of Eastern Europe (約翰 · 康納利 (John Connelly) 著；... 
血色大地：夾在希特勒與史達林之間的東歐 = Bloodlands Europe Between Hitler and Stalin (提摩希 · 史奈德（Timothy Snyder）著；陳榮彬, 劉維人 譯) 
血色大地：夹在希特勒与史达林之间的东欧 = Bloodlands Europe Between Hitler and Stalin (提摩希 · 史奈德（Timothy Snyder）著；陈荣彬, 刘维人 译) (t2c)
血色大地：夹在希特勒与斯大林之间的东欧 = Bloodlands Europe Between Hitler and Stalin (提摩希 · 史奈德（Timothy Snyder）著；陈荣彬, 刘维人 译) (t2c)
The Russian Revolution - A New History (2017) (Sean McMeekin) 
【東歐百年史 · 冊3】共同體的神話：柏林圍牆的倒塌與意識形態的洗牌 = From Peoples into Nations A History of Eastern Europe (約翰 · 康納利 (John Connelly) 著；... 
铁幕降临 ：赤色浪潮下的东欧 = Iron Curtain The Crushing of Eastern Europe, 1944-1956 (安爱波邦 (Anne Applebaum) 著  张葳 译) (sc)
抵制与反抗来自东欧的教训 (剑桥·公共安全管理译丛) (罗杰·D·彼得森) (z-library.sk, 1lib.sk, z-lib.sk)
铁幕降临赤色浪潮下的东欧 (安妮-阿普尔鲍姆Anne Applebaum) 
鐵幕降臨 ：赤色浪潮下的東歐 = Iron Curtain The Crushing of Eastern Europe, 1944-1956 (安愛波邦 (Anne Applebaum) 著  張葳 譯) 
俄国人 下册 (赫德里克·史密斯, 上海《国际问题资料》编辑组译) 
共同體的神話 = From Peoples into Nations A History of Eastern Europe (約翰 · 康納利 (John Connelly) 著  羅亞琪, 黃妤萱, 楊雅筑 etc.) 
東歐各國共產黨 (斯蒂芬·蓋拉蒂) 
罗马帝国的兴盛与衰落（一套尽览罗马帝国的兴亡更迭，全方位、多角度探秘罗马社会！套装共3册。） (汗青堂系列) (玛丽·比尔德  凯尔·哈珀) 
俄国人 上册 (赫德里克·史密斯, 上海《国际问题资料》编辑组译) 
大汗之国：西方眼中的中国 (史景迁作品) ( etc.) 
英格兰简史（英国历史学家詹金斯爵士潜心力作，泰晤士报畅销书，记录公元410年到21世纪的帝国兴衰，豆瓣9.7分精彩巨作） (［英］西蒙·詹金斯 [［... 
美国人的历史(套装共3册) (保罗·约翰逊) 
穿越百年中东 (郭建龙)";
    let documents: Vec<&str> = txt.lines().collect();

    let mut results = model.rerank("与东欧历史相关的书籍", documents, true, None)?;

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for rr in results {
        println!("{}, {}, {}", rr.index, rr.document.unwrap(), rr.score);
    }
    Ok(())
}
