use anyhow::Result;
use aphelios_search::SearchEngine;

#[test]
fn embedding_search_test() -> Result<()> {
    let mut se = SearchEngine::new()?;

    let file_list = vec![
        "二手时间".into(),         //
        "苏联简史".into(),         //
        "切尔诺贝利的悲鸣".into(), //
        "中国文学史".into(),
        "论人生短暂".into(),
        "地图在动".into(),
        "苏联的最后一天".into(), //
        "精神帝国 殖民历史与当代政治".into(),
        "传播与帝国 1860-1930年的媒体、市场与全球化".into(),
        "消解人民 新自由主義的寧靜革命".into(),
        "國家: 過去.現在.未來".into(),
        "中国禁忌简史".into(),
        "中国历代政治得失".into(),
        "一句顶一万句".into(),
    ];

    se.build_index(file_list)?;

    println!("\n搜索关键词: '苏联'");
    let results = se.hybrid_search("苏联", 5)?;

    for (name, score, label) in results {
        println!("[{}] {} (得分: {:.4})", label, name, score);
    }

    Ok(())
}

#[test]
fn hybrid_search_test() -> Result<()> {
    let mut se = SearchEngine::new()?;

    let file_list = vec![
        "二手时间".into(),         // 俄罗斯相关
        "苏联简史".into(),         // 苏联相关
        "切尔诺贝利的悲鸣".into(), // 乌克兰/苏联相关
        "中国文学史".into(),       // 中国文学
        "论人生短暂".into(),       // 哲学
        "地图在动".into(),         // 地理
        "苏联的最后一天".into(),   // 苏联相关
        "精神帝国 殖民历史与当代政治".into(), // 政治历史
        "传播与帝国 1860-1930年的媒体、市场与全球化".into(), // 历史
        "消解人民 新自由主義的寧靜革命".into(), // 政治经济
        "國家: 過去.現在.未來".into(), // 政治
        "中国禁忌简史".into(),     // 中国历史
        "中国历代政治得失".into(), // 中国政治
        "一句顶一万句".into(),     // 中国文学
        "红楼梦".into(),           // 中国古典文学
        "西游记".into(),           // 中国古典文学
        "三国演义".into(),         // 中国古典文学
        "水浒传".into(),           // 中国古典文学
        "资治通鉴".into(),         // 中国历史
        "史记".into(),             // 中国历史
        "毛泽东传".into(),         // 中国现代史
        "邓小平时代".into(),       // 中国现代史
        "习近平治国理政".into(),   // 中国当代政治
        "中华民国史".into(),       // 中国近代史
        "大清十二帝".into(),       // 中国清代史
    ];

    se.build_index(file_list)?;

    println!("\n=== 混合搜索测试 ===");
    
    // 测试1: 搜索苏联相关
    println!("\n搜索关键词: '苏联'");
    let results = se.hybrid_search("苏联", 5)?;
    for (name, score, label) in results {
        println!("[{}] {} (得分: {:.4})", label, name, score);
    }

    // 测试2: 搜索中国相关
    println!("\n搜索关键词: '中国'");
    let results = se.hybrid_search("中国", 5)?;
    for (name, score, label) in results {
        println!("[{}] {} (得分: {:.4})", label, name, score);
    }

    // 测试3: 搜索四大名著
    println!("\n搜索关键词: '四大名著'");
    let results = se.hybrid_search("四大名著", 5)?;
    for (name, score, label) in results {
        println!("[{}] {} (得分: {:.4})", label, name, score);
    }

    // 测试4: 搜索历史相关
    println!("\n搜索关键词: '历史'");
    let results = se.hybrid_search("历史", 5)?;
    for (name, score, label) in results {
        println!("[{}] {} (得分: {:.4})", label, name, score);
    }

    Ok(())
}