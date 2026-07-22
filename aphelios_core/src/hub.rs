use hf_hub::HFClientSync;
use once_cell::sync::OnceCell;
use std::{env, path::PathBuf};

static APHELIOS_MODEL_DIR: OnceCell<Option<String>> = OnceCell::new();
static CLIENT: OnceCell<HFClientSync> = OnceCell::new();

fn client() -> &'static HFClientSync {
    CLIENT.get_or_init(|| HFClientSync::new().expect("Failed to create HF client"))
}

pub fn get_aphelios_model_dir() -> Option<&'static str> {
    APHELIOS_MODEL_DIR
        .get_or_init(|| env::var("APHELIOS_MODEL_DIR").ok())
        .as_deref()
}

pub fn load_or_download(m_id: &str, model_id: Option<impl Into<String>>, file_name: &str) -> PathBuf {
    let m_i = model_id
        .map(|m| m.into())
        .unwrap_or_else(|| m_id.to_string());
    load_file_local_or_download(m_i, file_name)
}

pub fn load_file_local_or_download(model_id: impl AsRef<str>, file_name: &str) -> PathBuf {
    let id = model_id.as_ref();
    if id.starts_with("/") && PathBuf::from(id).exists() {
        PathBuf::from(id).join(file_name)
    } else {
        let (owner, name) = hf_hub::split_id(id);
        client()
            .model(owner, name)
            .download_file()
            .filename(file_name)
            .revision("main")
            .send()
            .expect("Failed to download model file")
    }
}
