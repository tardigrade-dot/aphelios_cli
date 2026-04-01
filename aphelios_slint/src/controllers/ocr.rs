use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use slint::{SharedString, VecModel};
use anyhow::Result;
use tracing::info;

use crate::controllers::AppContext;

pub struct OcrLogic {
    ctx: Arc<AppContext>,
    stop_flag: Arc<AtomicBool>,
}

impl OcrLogic {
    pub fn new(ctx: Arc<AppContext>) -> Self {
        Self {
            ctx,
            stop_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn ctx(&self) -> &Arc<AppContext> {
        &self.ctx
    }

    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::Relaxed);
    }

    pub fn start_ocr(
        &self, 
        model_path: String, 
        input_file: String, 
        output_dir: String,
        _log_model: Rc<VecModel<SharedString>>,
        on_complete: impl Fn(Result<Vec<String>>) + Send + 'static
    ) {
        self.stop_flag.store(false, Ordering::Relaxed);
        
        // 更新设置
        let mut settings = self.ctx.get_settings();
        settings.ocr_model_path = Some(model_path.clone());
        settings.ocr_output_dir = Some(output_dir.clone());
        self.ctx.save_settings(settings);

        let engine = self.ctx.ocr_engine.clone();
        let _stop_flag = self.stop_flag.clone();
        
        std::thread::spawn(move || {
            info!("Starting OCR: input={}, output={}, model={}", input_file, output_dir, model_path);
            
            let result = {
                let mut engine_guard = engine.lock().unwrap();
                engine_guard.dolphin_ocr(&model_path, &input_file, &output_dir)
            };
            
            on_complete(result);
        });
    }
}
