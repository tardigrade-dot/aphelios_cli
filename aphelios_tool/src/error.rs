use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid config")]
    InvalidConfig,

    #[error("anyhow error: {0}")]
    Anyhow(#[from] anyhow::Error),
}
