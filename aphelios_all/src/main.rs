use anyhow::Result;
use aphelios_cli::{cli::Cli, run};
use clap::Parser;
use std::{thread, time::Duration};

#[tokio::main]
async fn main() -> Result<()> {
    print_banner();
    run_cli().await;
    Ok(())
}

pub fn print_banner() {
    // ANSI 颜色
    const PURPLE: &str = "\x1b[95m";
    const CYAN: &str = "\x1b[96m";
    const RESET: &str = "\x1b[0m";
    const BOLD: &str = "\x1b[1m";

    let banner = r#"
 █████╗ ██████╗ ██╗  ██╗███████╗██╗     ██╗ ██████╗ ███████╗
██╔══██╗██╔══██╗██║  ██║██╔════╝██║     ██║██╔═══██╗██╔════╝
███████║██████╔╝███████║█████╗  ██║     ██║██║   ██║███████╗
██╔══██║██╔═══╝ ██╔══██║██╔══╝  ██║     ██║██║   ██║╚════██║
██║  ██║██║     ██║  ██║███████╗███████╗██║╚██████╔╝███████║
╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝ ╚═════╝ ╚══════╝
"#;

    println!("{BOLD}{PURPLE}{banner}{RESET}");
    println!("{CYAN}{BOLD}              aphelios_cli  •  OCR Engine{RESET}");
    println!("{CYAN}              Fast • Async • Stream-based OCR{RESET}");
    println!();
}

async fn run_cli() {
    let cli = Cli::parse();

    if let Err(err) = run(cli).await {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
