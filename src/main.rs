use aphelios_cli::{cli::Cli, run};
use clap::Parser;

#[tokio::main]
async fn main() {
    println!("Hello, world!");

    let cli = Cli::parse();

    if let Err(err) = run(cli).await {
        eprintln!("{err}");
        std::process::exit(1);
    }

}
