use jieba_rs::Jieba;
use zhconv::{zhconv, Variant};

#[test]
fn test() {
    assert_eq!(zhconv("雾失楼台，月迷津渡", Variant::ZhTW), "霧失樓臺，月迷津渡");
    assert_eq!(zhconv("驛寄梅花，魚傳尺素", "zh-Hans".parse().unwrap()), "驿寄梅花，鱼传尺素");
    println!("{}", zhconv("在這座城市內生活著名為XX的種族", Variant::ZhCN));
}

#[test]
fn test2() {
    let jieba = Jieba::new();
    let words = jieba.cut("我们中出了一个叛徒", false);
    assert_eq!(
        words
            .iter()
            .map(|s| s.word)
            .collect::<Vec<&str>>(),
        vec!["我们", "中", "出", "了", "一个", "叛徒"]
    );

    let words2 = jieba.cut("在這座城市內生活著名為XX的種族", true);
    let _ = words2
        .iter()
        .map(|s| {
            println!("{}", s.word);
            s.word
        })
        .collect::<Vec<&str>>();
}

#[test]
fn test3() {
    let a = "hello";
    println!("{}", a);
    println!("{a}");
    // println!(a);
    println!("hello");

}
