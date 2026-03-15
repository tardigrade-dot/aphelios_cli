import React, { useState, useCallback, useRef } from 'react';
import JSZip from 'jszip';
import * as OpenCC from 'opencc-js';
import { saveAs } from 'file-saver';
import { Upload, FileText, Download, CheckCircle2, Loader2, AlertCircle, BookOpen, ArrowRightLeft, Settings, Brain } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

type ConversionStatus = 'idle' | 'processing' | 'completed' | 'error';
type ConversionMode = 't2s' | 's2t';

const DEFAULT_DISPUTED_WORDS = '著,發,樂,後,覺,乾,裡,會,髮,費,對,標,愛,麵,裝,視,廣,龍,雲,鳥,機,藝,書,雜,錢,頭,錄,顏,體,龜,魚';

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<ConversionStatus>('idle');
  const [mode, setMode] = useState<ConversionMode>('t2s');
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  
  // Strong Conversion States
  const [useStrongConversion, setUseStrongConversion] = useState(false);
  const [disputedWords, setDisputedWords] = useState(DEFAULT_DISPUTED_WORDS);
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const unmaskerRef = useRef<any>(null);

  const loadModel = async () => {
    try {
      setIsModelLoading(true);
      setError(null);
      const { pipeline, env } = await import('@huggingface/transformers');
      // env.remoteHost = 'https://modelscope.cn';
      // 强制从 jsdelivr 获取 wasm，因为我们已经在 vite 中 external 了它
      env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
      
      unmaskerRef.current = await pipeline('fill-mask', 'Xenova/bert-base-chinese', {
        device: 'webgpu',
        dtype: 'fp16',
        progress_callback: (p: any) => {
          if (p.status === 'progress') {
            setProgress(Math.floor(p.progress));
          }
        },
      });
      
      setModelLoaded(true);
      setIsModelLoading(false);
      setProgress(0);
    } catch (err) {
      console.error('Failed to load model:', err);
      setError('模型加载失败，请检查网络连接。');
      setIsModelLoading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (selectedFile.name.toLowerCase().endsWith('.epub')) {
        setFile(selectedFile);
        setError(null);
        setStatus('idle');
        setProgress(0);
      } else {
        setError('请上传有效的 .epub 文件');
      }
    }
  };

  const processWithMask = async (text: string) => {
    if (!unmaskerRef.current || !useStrongConversion) return text;

    const words = disputedWords.split(/[,，\s]+/).filter(w => w.length > 0);
    if (words.length === 0) return text;

    let resultText = text;
    // We process in chunks to avoid blocking the UI too much and handle large texts
    // For simplicity, we search for each disputed word
    for (const word of words) {
      let index = resultText.indexOf(word);
      while (index !== -1) {
        // Extract context (e.g., 20 chars around)
        const start = Math.max(0, index - 20);
        const end = Math.min(resultText.length, index + word.length + 20);
        const context = resultText.substring(start, end);
        const maskedContext = context.replace(word, '[MASK]');
        
        try {
          const predictions = await unmaskerRef.current(maskedContext);
          if (predictions && predictions.length > 0) {
            const topPrediction = predictions[0].token_str;
            if (topPrediction && topPrediction !== '[UNK]') {
              // Replace only the specific occurrence
              resultText = resultText.substring(0, index) + topPrediction + resultText.substring(index + word.length);
            }
          }
        } catch (err) {
          console.warn('Mask prediction failed for context:', maskedContext, err);
        }
        
        index = resultText.indexOf(word, index + 1);
      }
    }
    return resultText;
  };

  const convertEpub = async () => {
    if (!file) return;
    if (useStrongConversion && !modelLoaded) {
      setError('请先加载 MASK 模型以启用强转换功能。');
      return;
    }

    try {
      setStatus('processing');
      setProgress(5);
      setError(null);

      const zip = new JSZip();
      const content = await zip.loadAsync(file);
      
      const converter = OpenCC.Converter({ 
        from: mode === 't2s' ? 'hk' : 'cn', 
        to: mode === 't2s' ? 'cn' : 'hk' 
      });
      
      const fileNames = Object.keys(content.files);
      const totalFiles = fileNames.length;
      let processedCount = 0;

      for (const fileName of fileNames) {
        const zipFile = content.files[fileName];
        
        if (!zipFile.dir && (
          fileName.endsWith('.html') || 
          fileName.endsWith('.xhtml') || 
          fileName.endsWith('.opf') || 
          fileName.endsWith('.ncx') ||
          fileName.endsWith('.txt')
        )) {
          const text = await zipFile.async('string');
          // Step 1: OpenCC conversion
          let convertedText = converter(text);
          
          // Step 2: MASK model processing (Strong Conversion)
          if (useStrongConversion) {
            convertedText = await processWithMask(convertedText);
          }
          
          zip.file(fileName, convertedText);
        }
        
        processedCount++;
        setProgress(Math.floor(10 + (processedCount / totalFiles) * 80));
      }

      setProgress(95);
      const blob = await zip.generateAsync({ type: 'blob', mimeType: 'application/epub+zip' });
      
      const suffix = mode === 't2s' ? '_简体' : '_繁体';
      const newFileName = file.name.replace(/\.epub$/i, `${suffix}.epub`);
      saveAs(blob, newFileName);
      
      setProgress(100);
      setStatus('completed');
    } catch (err) {
      console.error('Conversion failed:', err);
      setError('转换失败，请确保文件格式正确且未加密。');
      setStatus('error');
    }
  };

  const reset = () => {
    setFile(null);
    setStatus('idle');
    setProgress(0);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-[#f5f5f5] text-[#1a1a1a] font-sans selection:bg-emerald-100">
      <div className="max-w-3xl mx-auto px-6 py-12 md:py-24">
        {/* Header */}
        <header className="mb-12 text-center md:text-left">
          <div className="flex items-center justify-center md:justify-start gap-3 mb-4">
            <div className="p-2 bg-emerald-600 rounded-xl text-white shadow-lg shadow-emerald-200">
              <BookOpen size={32} />
            </div>
            <h1 className="text-4xl font-bold tracking-tight">EPUB 繁简转换</h1>
          </div>
          <p className="text-lg text-zinc-500 max-w-xl">
            上传您的 EPUB 电子书，我们将为您快速进行繁简转换。
            所有处理均在您的浏览器中完成，保护您的隐私。
          </p>
        </header>

        <main>
          <div className="bg-white rounded-3xl shadow-sm border border-zinc-100 p-8 md:p-12">
            opencc后置处理
          </div>
          <div className="bg-white rounded-3xl shadow-sm border border-zinc-100 p-8 md:p-12">
            {/* Mode Selector */}
            {status === 'idle' && (
              <div className="space-y-6 mb-10">
                <div className="flex justify-center">
                  <div className="bg-zinc-100 p-1 rounded-2xl flex gap-1">
                    <button
                      onClick={() => setMode('t2s')}
                      className={`px-6 py-2.5 rounded-xl text-sm font-bold transition-all ${
                        mode === 't2s' 
                          ? 'bg-white text-emerald-600 shadow-sm' 
                          : 'text-zinc-500 hover:text-zinc-800'
                      }`}
                    >
                      繁体 → 简体
                    </button>
                    <button
                      onClick={() => setMode('s2t')}
                      className={`px-6 py-2.5 rounded-xl text-sm font-bold transition-all ${
                        mode === 's2t' 
                          ? 'bg-white text-emerald-600 shadow-sm' 
                          : 'text-zinc-500 hover:text-zinc-800'
                      }`}
                    >
                      简体 → 繁体
                    </button>
                  </div>
                </div>

                {/* Strong Conversion Options */}
                <div className="bg-zinc-50 rounded-2xl p-6 border border-zinc-100 space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Brain className="text-emerald-600" size={20} />
                      <span className="font-bold text-sm">强转换选项 (AI 纠错)</span>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input 
                        type="checkbox" 
                        className="sr-only peer"
                        checked={useStrongConversion}
                        onChange={(e) => setUseStrongConversion(e.target.checked)}
                      />
                      <div className="w-11 h-6 bg-zinc-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-zinc-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-emerald-600"></div>
                    </label>
                  </div>
                  
                  {useStrongConversion && (
                    <motion.div 
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      className="space-y-4 pt-2 border-t border-zinc-200"
                    >
                      <div className="flex items-center justify-between gap-4">
                        <p className="text-xs text-zinc-500 flex-1">
                          启用 BERT MASK 模型对争议字进行上下文推理。初次使用需下载约 400MB 模型。
                        </p>
                        <button
                          onClick={loadModel}
                          disabled={modelLoaded || isModelLoading}
                          className={`
                            px-4 py-2 rounded-xl text-xs font-bold whitespace-nowrap transition-all
                            ${modelLoaded 
                              ? 'bg-emerald-100 text-emerald-700 cursor-default' 
                              : isModelLoading 
                                ? 'bg-zinc-200 text-zinc-400 cursor-wait'
                                : 'bg-zinc-900 text-white hover:bg-zinc-800'}
                          `}
                        >
                          {modelLoaded ? '模型已就绪' : isModelLoading ? '正在加载...' : '加载 AI 模型'}
                        </button>
                      </div>
                      
                      <div className="space-y-2">
                        <label className="text-xs font-bold text-zinc-600 flex items-center gap-1">
                          <Settings size={14} /> 自定义争议字 (逗号分隔)
                        </label>
                        <textarea
                          value={disputedWords}
                          onChange={(e) => setDisputedWords(e.target.value)}
                          placeholder="例如: 著,發,樂..."
                          className="w-full h-20 p-3 rounded-xl border border-zinc-200 text-xs focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none transition-all resize-none bg-white"
                        />
                      </div>
                    </motion.div>
                  )}
                </div>
              </div>
            )}

            <AnimatePresence mode="wait">
              {status === 'idle' && (
                <motion.div
                  key="idle"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="space-y-8"
                >
                  <div 
                    className={`
                      relative border-2 border-dashed rounded-2xl p-12 text-center transition-all
                      ${file ? 'border-emerald-500 bg-emerald-50/30' : 'border-zinc-200 hover:border-emerald-400 hover:bg-zinc-50'}
                    `}
                  >
                    <input
                      type="file"
                      accept=".epub"
                      onChange={handleFileChange}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    />
                    <div className="flex flex-col items-center gap-4">
                      <div className={`p-4 rounded-full ${file ? 'bg-emerald-100 text-emerald-600' : 'bg-zinc-100 text-zinc-400'}`}>
                        {file ? <FileText size={40} /> : <Upload size={40} />}
                      </div>
                      <div>
                        <p className="text-lg font-medium">
                          {file ? file.name : '点击或拖拽 EPUB 文件到这里'}
                        </p>
                        <p className="text-sm text-zinc-400 mt-1">
                          支持 .epub 格式，建议文件大小不超过 50MB
                        </p>
                      </div>
                    </div>
                  </div>

                  {error && (
                    <div className="flex items-center gap-2 text-red-500 bg-red-50 p-4 rounded-xl border border-red-100">
                      <AlertCircle size={20} />
                      <p className="text-sm font-medium">{error}</p>
                    </div>
                  )}

                  <button
                    onClick={convertEpub}
                    disabled={!file || (useStrongConversion && !modelLoaded)}
                    className={`
                      w-full py-4 rounded-2xl font-bold text-lg transition-all flex items-center justify-center gap-2
                      ${file && (!useStrongConversion || modelLoaded)
                        ? 'bg-emerald-600 text-white hover:bg-emerald-700 shadow-lg shadow-emerald-100 active:scale-[0.98]' 
                        : 'bg-zinc-100 text-zinc-400 cursor-not-allowed'}
                    `}
                  >
                    {isModelLoading ? `正在加载模型 ${progress}%` : `开始${mode === 't2s' ? '繁转简' : '简转繁'}${useStrongConversion ? ' (强转换)' : ''}`}
                  </button>
                </motion.div>
              )}

              {status === 'processing' && (
                <motion.div
                  key="processing"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="py-12 text-center space-y-8"
                >
                  <div className="relative inline-block">
                    <Loader2 size={64} className="text-emerald-600 animate-spin" />
                    <div className="absolute inset-0 flex items-center justify-center text-xs font-bold text-emerald-700">
                      {progress}%
                    </div>
                  </div>
                  <div className="space-y-4">
                    <h2 className="text-2xl font-bold">正在转换中...</h2>
                    <p className="text-zinc-500">正在进行{mode === 't2s' ? '繁转简' : '简转繁'}转换{useStrongConversion ? '及 AI 纠错' : ''}，请稍候</p>
                    <div className="w-full bg-zinc-100 h-2 rounded-full overflow-hidden max-w-md mx-auto">
                      <motion.div 
                        className="h-full bg-emerald-600"
                        initial={{ width: 0 }}
                        animate={{ width: `${progress}%` }}
                        transition={{ duration: 0.3 }}
                      />
                    </div>
                  </div>
                </motion.div>
              )}

              {status === 'completed' && (
                <motion.div
                  key="completed"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="py-12 text-center space-y-8"
                >
                  <div className="flex justify-center">
                    <div className="p-6 bg-emerald-100 text-emerald-600 rounded-full">
                      <CheckCircle2 size={64} />
                    </div>
                  </div>
                  <div className="space-y-4">
                    <h2 className="text-2xl font-bold">转换成功！</h2>
                    <p className="text-zinc-500">您的{mode === 't2s' ? '简体' : '繁体'}版 EPUB 已准备就绪。</p>
                  </div>
                  <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
                    <button
                      onClick={reset}
                      className="px-8 py-4 rounded-2xl border border-zinc-200 font-bold hover:bg-zinc-50 transition-all"
                    >
                      转换另一个
                    </button>
                    <button
                      onClick={() => {
                        setError('下载已开始');
                        setTimeout(() => setError(null), 3000);
                      }}
                      className="px-8 py-4 rounded-2xl bg-emerald-600 text-white font-bold hover:bg-emerald-700 shadow-lg shadow-emerald-100 transition-all flex items-center justify-center gap-2"
                    >
                      <Download size={20} />
                      重新下载
                    </button>
                  </div>
                </motion.div>
              )}

              {status === 'error' && (
                <motion.div
                  key="error"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="py-12 text-center space-y-8"
                >
                  <div className="flex justify-center">
                    <div className="p-6 bg-red-100 text-red-600 rounded-full">
                      <AlertCircle size={64} />
                    </div>
                  </div>
                  <div className="space-y-4">
                    <h2 className="text-2xl font-bold text-red-600">出错了</h2>
                    <p className="text-zinc-500">{error || '转换过程中发生未知错误'}</p>
                  </div>
                  <button
                    onClick={reset}
                    className="px-8 py-4 rounded-2xl bg-zinc-900 text-white font-bold hover:bg-zinc-800 transition-all"
                  >
                    返回重试
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Features */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
            <div className="p-6 bg-white rounded-2xl border border-zinc-100 shadow-sm">
              <h3 className="font-bold mb-2">隐私安全</h3>
              <p className="text-sm text-zinc-500">所有转换均在本地浏览器完成，您的书籍不会被上传到任何服务器。</p>
            </div>
            <div className="p-6 bg-white rounded-2xl border border-zinc-100 shadow-sm">
              <h3 className="font-bold mb-2">智能纠错</h3>
              <p className="text-sm text-zinc-500">可选启用 BERT MASK 模型，自动修复繁简转换中常见的歧义词。</p>
            </div>
            <div className="p-6 bg-white rounded-2xl border border-zinc-100 shadow-sm">
              <h3 className="font-bold mb-2">双向转换</h3>
              <p className="text-sm text-zinc-500">支持繁体到简体以及简体到繁体的双向智能转换，满足不同阅读需求。</p>
            </div>
          </div>
        </main>

        <footer className="mt-24 text-center text-sm text-zinc-400">
          <p>© {new Date().getFullYear()} EPUB 繁简转换工具 · 纯净无广告</p>
        </footer>
      </div>
    </div>
  );
}
