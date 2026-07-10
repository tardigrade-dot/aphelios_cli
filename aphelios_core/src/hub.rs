use std::{env, path::PathBuf};
use hf_hub::api::sync::{Api, ApiBuilder};
use once_cell::sync::OnceCell;

static APHELIOS_MODEL_DIR: OnceCell<Option<String>> = OnceCell::new();
static API: OnceCell<Api> = OnceCell::new();

pub fn get_aphelios_model_dir() -> Option<&'static str> {
    APHELIOS_MODEL_DIR
        .get_or_init(|| env::var("APHELIOS_MODEL_DIR").ok())
        .as_deref()
}

pub fn load_or_download(m_id: &str, model_id:  Option<impl Into<String>>, file_name: &str) -> PathBuf{
    let m_i = model_id
                .map(|m| m.into())
                .unwrap_or_else(|| m_id.to_string());
    load_file_local_or_download(m_i, file_name)
}

pub fn load_file_local_or_download(model_id: impl AsRef<str>, file_name: &str) -> PathBuf{
    if model_id.as_ref().starts_with("/") && PathBuf::from(model_id.as_ref()).exists() {
        PathBuf::from(model_id.as_ref()).join(file_name)
    }else {
        let model_repo = API.get_or_init(|| {
            ApiBuilder::new()
                .with_progress(true)
                .build()
                .unwrap()
        }).repo(hf_hub::Repo::with_revision(
            model_id.as_ref().to_string(),
            hf_hub::RepoType::Model,
            "main".to_string(),
        ));

        model_repo.get(file_name).unwrap()
    }
}
