use crate::error::Result;
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use std::time::Duration;
use tokio::time::sleep;

pub async fn run(verbose: bool) -> Result<()> {
    if verbose {
        println!("running in verbose mode");
    }
    // 随机生成耗时，0~1000ms
    let mut rng = rand::rng();
    let total_ms = rng.random_range(200..=1000);

    // 创建进度条
    let pb = ProgressBar::new(1000);
    pb.set_style(
        ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {percent}%")
            .unwrap()
            .progress_chars("#>-"),
    );

    // 每 10ms 更新一次
    let mut elapsed = 0;
    while elapsed < total_ms {
        sleep(Duration::from_millis(rng.random_range(200..=1000))).await;
        elapsed += 20;
        if elapsed > total_ms {
            elapsed = total_ms;
        }
        pb.set_position(elapsed);
    }

    pb.finish_with_message("Done!");

    if verbose {
        println!("Completed random sleep of {} ms", total_ms);
    }
    Ok(())
}
