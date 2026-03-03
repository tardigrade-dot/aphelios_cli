// 包含所有生成的 Slint UI 文件
include!(concat!(env!("OUT_DIR"), "/main_menu.rs"));
include!(concat!(env!("OUT_DIR"), "/ocr_ui.rs"));
include!(concat!(env!("OUT_DIR"), "/asr_ui.rs"));
include!(concat!(env!("OUT_DIR"), "/tts_ui.rs"));

use anyhow::Result;
use aphelios_core::{utils::core_utils, AudioLoader};
use aphelios_ocr::dolphin::model::DolphinModel;
use aphelios_tts::qwen_tts::qwen3_tts_with_output;
use slint::{ComponentHandle, Model, ModelRc, SharedString, VecModel};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tracing::{error, info};

// 全局音频路径，用于播放
static AUDIO_OUTPUT_PATH: Mutex<Option<String>> = Mutex::new(None);

// 辅助函数：向日志添加消息
fn add_log_message(log_model: &Rc<VecModel<SharedString>>, msg: impl Into<SharedString>) {
    log_model.push(msg.into());
}

fn main() -> Result<()> {
    core_utils::init_tracing();

    let main_menu = MainMenu::new()?;
    let main_menu_weak = main_menu.as_weak();

    main_menu.on_open_ocr({
        let w = main_menu_weak.clone();
        move || {
            let _ = w.upgrade();
            let _ = run_ocr_ui();
        }
    });

    main_menu.on_open_asr({
        let w = main_menu_weak.clone();
        move || {
            let _ = w.upgrade();
            let _ = run_asr_ui();
        }
    });

    main_menu.on_open_tts({
        let w = main_menu_weak.clone();
        move || {
            let _ = w.upgrade();
            let _ = run_tts_ui();
        }
    });

    main_menu.run()?;
    Ok(())
}

fn run_ocr_ui() -> Result<()> {
    let window = OcrWindow::new()?;
    window.set_model_path(
        "/Volumes/sw/pretrained_models/Dolphin-v1.5"
            .to_string()
            .into(),
    );
    let window_weak = window.as_weak();

    window.on_go_back({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let _: Result<(), _> = win.hide();
            }
        }
    });

    window.on_select_input_file({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .add_filter("图片与 PDF", &["png", "jpg", "jpeg", "pdf", "tiff", "bmp"])
                    .pick_file()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_input_file_path(path.into());
            }
        }
    });

    window.on_select_output_dir({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .pick_folder()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_output_dir_path(path.into());
            }
        }
    });

    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_flag_clone = stop_flag.clone();
    let log_model: Rc<VecModel<SharedString>> = Rc::new(VecModel::default());
    window.set_log_messages(ModelRc::from(log_model.clone()));

    window.on_start_ocr({
        let w = window_weak.clone();
        let sf = stop_flag_clone.clone();
        let lm = log_model.clone();
        move || {
            sf.store(false, Ordering::Relaxed);
            let Some(win) = w.upgrade() else { return };

            let input_file: String = win.get_input_file_path().to_string();
            let output_dir: String = win.get_output_dir_path().to_string();
            let model_path: String = win.get_model_path().to_string();

            win.set_is_running(true);
            win.set_status_message("OCR 执行中...".into());
            add_log_message(&lm, format!("开始 OCR: {}", input_file));
            add_log_message(&lm, format!("输出目录：{}", output_dir));
            add_log_message(&lm, format!("模型路径：{}", model_path));

            let w2 = w.clone();
            let w3 = w.clone();
            std::thread::spawn(move || {
                info!(
                    "Starting OCR: input={}, output={}, model={}",
                    input_file, output_dir, model_path
                );
                let result = match DolphinModel::load_model(&model_path) {
                    Ok(mut dm) => {
                        let rt = tokio::runtime::Builder::new_current_thread()
                            .enable_all()
                            .build()
                            .unwrap();
                        rt.block_on(dm.dolphin_ocr(&input_file, &output_dir))
                    }
                    Err(e) => {
                        error!("Failed to load model: {}", e);
                        Err(e)
                    }
                };

                match result {
                    Ok(results) => {
                        info!("OCR completed with {} results", results.len());
                        let results_clone = results.clone();
                        let _ = slint::invoke_from_event_loop(move || {
                            let Some(win) = w2.upgrade() else { return };
                            let log_model: ModelRc<SharedString> = win.get_log_messages();
                            if let Some(vm) =
                                log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                            {
                                vm.push(
                                    format!("OCR 完成！识别出 {} 条结果", results_clone.len())
                                        .into(),
                                );
                                for (i, result) in results_clone.iter().take(10).enumerate() {
                                    vm.push(format!("[{}] {}", i + 1, result).into());
                                }
                                if results_clone.len() > 10 {
                                    vm.push(
                                        format!("... 还有 {} 条结果", results_clone.len() - 10)
                                            .into(),
                                    );
                                }
                            }
                            win.set_status_message("OCR 完成!".into());
                            win.set_is_running(false);
                        });
                    }
                    Err(e) => {
                        error!("OCR failed: {}", e);
                        let error_msg: String = e.to_string();
                        let _ = slint::invoke_from_event_loop(move || {
                            let Some(win) = w3.upgrade() else { return };
                            let log_model: ModelRc<SharedString> = win.get_log_messages();
                            if let Some(vm) =
                                log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                            {
                                vm.push(format!("OCR 失败：{}", error_msg).into());
                            }
                            win.set_status_message("OCR 失败".into());
                            win.set_is_running(false);
                        });
                    }
                }
            });
        }
    });

    window.on_stop_ocr({
        let w = window_weak.clone();
        let lm = log_model.clone();
        move || {
            stop_flag.store(true, Ordering::Relaxed);
            if let Some(win) = w.upgrade() {
                win.set_is_running(false);
                win.set_status_message("OCR 已停止".into());
                add_log_message(&lm, "OCR 已停止");
            }
        }
    });

    window.run()?;
    Ok(())
}

fn run_asr_ui() -> Result<()> {
    let window: AsrWindow = AsrWindow::new()?;
    let window_weak = window.as_weak();

    window.on_go_back({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let _: Result<(), slint::PlatformError> = win.hide();
            }
        }
    });

    window.on_select_audio_file({
        let w: slint::Weak<AsrWindow> = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .add_filter("音频文件", &["wav", "mp3", "flac", "ogg", "m4a"])
                    .pick_file()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_audio_file_path(path.into());
            }
        }
    });

    window.on_select_output_path({
        let w: slint::Weak<AsrWindow> = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .add_filter("文本文件", &["txt"])
                    .save_file()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_output_path(path.into());
            }
        }
    });

    let log_model: Rc<VecModel<SharedString>> = Rc::new(VecModel::default());
    window.set_log_messages(ModelRc::from(log_model.clone()));

    window.on_start_asr({
        let w: slint::Weak<AsrWindow> = window_weak.clone();
        move || {
            let Some(win) = w.upgrade() else { return };
            let audio_file: String = win.get_audio_file_path().to_string();
            let output_path: String = win.get_output_path().to_string();

            win.set_is_running(true);
            win.set_status_message("ASR 识别中...".into());

            let log_model: Rc<VecModel<SharedString>> = Rc::new(VecModel::default());
            win.set_log_messages(ModelRc::from(log_model.clone()));
            add_log_message(&log_model, format!("开始 ASR 识别：{}", audio_file));

            let w2 = w.clone();
            let w3 = w.clone();
            std::thread::spawn(move || {
                info!("Starting ASR: input={}", audio_file);
                use aphelios_core::utils::core_utils::PARAKEET_TDT_MODEL_PATH;
                use parakeet_rs::{ParakeetTDT, TimestampMode, Transcriber};

                let result = (|| -> Result<String> {
                    let mut parakeet = ParakeetTDT::from_pretrained(PARAKEET_TDT_MODEL_PATH, None)
                        .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;
                    let audio = AudioLoader::new().load(&audio_file)?.to_vec_f32();
                    Ok(parakeet
                        .transcribe_samples(audio, 16000, 1, Some(TimestampMode::Sentences))?
                        .text)
                })();

                match result {
                    Ok(text) => {
                        info!("ASR completed: {}", text);
                        let text_clone = text.clone();
                        let _ = slint::invoke_from_event_loop(move || {
                            let Some(win) = w2.upgrade() else { return };
                            win.set_transcription_result(text_clone.into());
                            win.set_status_message("ASR 完成!".into());
                            win.set_is_running(false);
                            let log_model: ModelRc<SharedString> = win.get_log_messages();
                            if let Some(vm) =
                                log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                            {
                                vm.push("ASR 识别完成!".into());
                            }
                        });
                    }
                    Err(e) => {
                        error!("ASR failed: {}", e);
                        let error_msg: String = e.to_string();
                        let _ = slint::invoke_from_event_loop(move || {
                            let Some(win) = w3.upgrade() else { return };
                            win.set_status_message("ASR 失败".into());
                            win.set_is_running(false);
                            let log_model: ModelRc<SharedString> = win.get_log_messages();
                            if let Some(vm) =
                                log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                            {
                                vm.push(format!("ASR 失败：{}", error_msg).into());
                            }
                        });
                    }
                }
            });
        }
    });

    window.run()?;
    Ok(())
}

fn run_tts_ui() -> Result<()> {
    let window = TtsWindow::new()?;
    window.set_model_path(
        "/Volumes/sw/pretrained_models/Qwen3-TTS-12Hz-0.6B-Base"
            .to_string()
            .into(),
    );
    let window_weak = window.as_weak();

    window.on_go_back({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let _: Result<(), _> = win.hide();
            }
        }
    });

    window.on_select_model_path({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .pick_folder()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_model_path(path.into());
            }
        }
    });

    window.on_select_output_path({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .add_filter("WAV 音频", &["wav"])
                    .save_file()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_output_path(path.into());
            }
        }
    });

    window.on_select_ref_audio({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .add_filter("音频文件", &["wav", "mp3", "flac"])
                    .pick_file()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_ref_audio_path(path.into());
            }
        }
    });

    let log_model: Rc<VecModel<SharedString>> = Rc::new(VecModel::default());
    window.set_log_messages(ModelRc::from(log_model.clone()));

    window.on_start_tts({
        let w = window_weak.clone();
        let lm = log_model.clone();
        move || {
            let Some(win) = w.upgrade() else { return };
            let input_text: String = win.get_input_text().to_string();
            let model_path: String = win.get_model_path().to_string();
            let output_path: String = win.get_output_path().to_string();
            let ref_audio_path: String = win.get_ref_audio_path().to_string();
            let ref_text: String = win.get_ref_text().to_string();

            win.set_is_running(true);
            win.set_status_message("TTS 合成中...".into());
            win.set_has_audio(false);
            add_log_message(&lm, format!("开始 TTS 合成：{}", input_text));
            add_log_message(&lm, format!("模型路径：{}", model_path));
            add_log_message(&lm, format!("输出路径：{}", output_path));

            let w_inner = w.clone();
            std::thread::spawn(move || {
                info!("Starting TTS: text={}, output={}", input_text, output_path);
                let result = qwen3_tts_with_output(
                    &input_text,
                    &model_path,
                    &output_path,
                    if ref_audio_path.is_empty() {
                        None
                    } else {
                        Some(&ref_audio_path)
                    },
                    if ref_text.is_empty() {
                        None
                    } else {
                        Some(&ref_text)
                    },
                );

                let w2 = w_inner.clone();
                let w3 = w_inner.clone();
                match result {
                    Ok(_) => {
                        info!("TTS completed: {}", output_path);
                        let _ = slint::invoke_from_event_loop(move || {
                            let Some(win) = w2.upgrade() else { return };
                            win.set_status_message("TTS 完成!".into());
                            win.set_is_running(false);
                            win.set_has_audio(true);
                            let log_model: ModelRc<SharedString> = win.get_log_messages();
                            if let Some(vm) =
                                log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                            {
                                vm.push("TTS 合成完成!".into());
                            }
                            if let Ok(mut path) = AUDIO_OUTPUT_PATH.lock() {
                                *path = Some(output_path.clone());
                            }
                        });
                    }
                    Err(e) => {
                        error!("TTS failed: {}", e);
                        let error_msg: String = e.to_string();
                        let _ = slint::invoke_from_event_loop(move || {
                            let Some(win) = w3.upgrade() else { return };
                            win.set_status_message("TTS 失败".into());
                            win.set_is_running(false);
                            let log_model: ModelRc<SharedString> = win.get_log_messages();
                            if let Some(vm) =
                                log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                            {
                                vm.push(format!("TTS 失败：{}", error_msg).into());
                            }
                        });
                    }
                }
            });
        }
    });

    window.on_play_audio({
        let w = window_weak.clone();
        let lm = log_model.clone();
        move || {
            let Some(_win) = w.upgrade() else { return };
            if let Ok(path_guard) = AUDIO_OUTPUT_PATH.lock() {
                if let Some(ref path) = *path_guard {
                    add_log_message(&lm, format!("播放音频：{}", path));
                    let path_clone = path.clone();
                    std::thread::spawn(move || {
                        play_audio_file(&path_clone);
                    });
                }
            }
        }
    });

    window.run()?;
    Ok(())
}

fn play_audio_file(path: &str) {
    use rodio::{OutputStream, Source};
    match (|| -> Result<()> {
        let (_stream, stream_handle) = OutputStream::try_default()?;
        let file = std::fs::File::open(path)?;
        let sink = rodio::Sink::try_new(&stream_handle)?;
        sink.append(rodio::Decoder::new(file)?);
        sink.sleep_until_end();
        Ok(())
    })() {
        Ok(_) => info!("Audio playback completed: {}", path),
        Err(e) => error!("Audio playback failed: {}", e),
    }
}
