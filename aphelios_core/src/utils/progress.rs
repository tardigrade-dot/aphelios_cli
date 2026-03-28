use std::sync::Arc;
use indicatif::ProgressBar;

/// A wrapper around `indicatif::ProgressBar` that also updates a UI callback.
#[derive(Clone)]
pub struct AppProgressBar {
    pb: ProgressBar,
    ui_callback: Option<Arc<dyn Fn(f32) + Send + Sync>>,
}

impl AppProgressBar {
    pub fn new(pb: ProgressBar) -> Self {
        Self {
            pb,
            ui_callback: None,
        }
    }

    pub fn with_ui(pb: ProgressBar, callback: impl Fn(f32) + Send + Sync + 'static) -> Self {
        Self {
            pb,
            ui_callback: Some(Arc::new(callback)),
        }
    }

    pub fn set_length(&self, len: u64) {
        self.pb.set_length(len);
    }

    pub fn set_position(&self, pos: u64) {
        self.pb.set_position(pos);
        if let Some(cb) = &self.ui_callback {
            let len = self.pb.length().unwrap_or(pos.max(1));
            cb(pos as f32 / len as f32);
        }
    }

    pub fn inc(&self, delta: u64) {
        self.pb.inc(delta);
        if let Some(cb) = &self.ui_callback {
            let pos = self.pb.position();
            let len = self.pb.length().unwrap_or(pos.max(1));
            cb(pos as f32 / len as f32);
        }
    }

    pub fn finish_with_message(&self, msg: impl Into<std::borrow::Cow<'static, str>>) {
        self.pb.finish_with_message(msg);
        if let Some(cb) = &self.ui_callback {
            cb(1.0);
        }
    }

    pub fn pb(&self) -> &ProgressBar {
        &self.pb
    }
}
