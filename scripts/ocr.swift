import Vision
import AppKit

func extractText(from imagePath: String) {
    guard let image = NSImage(contentsOfFile: imagePath),
          let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        print("无法加载图片")
        return
    }

    let request = VNRecognizeTextRequest { request, error in
        guard let observations = request.results as? [VNRecognizedTextObservation] else {
            print("识别失败")
            return
        }

        let text = observations.compactMap { observation in
            observation.topCandidates(1).first?.string
        }.joined(separator: "\n")

        print(text)
    }

    request.recognitionLevel = .accurate
    request.recognitionLanguages = ["zh-Hans", "en-US"]
    request.usesLanguageCorrection = true

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    try? handler.perform([request])
}

// 从命令行参数获取图片路径
if CommandLine.arguments.count > 1 {
    extractText(from: CommandLine.arguments[1])
} else {
    print("用法: swift ocr.swift <图片路径> \n example: swift /path/to/ocr.swift /path/to/page_32.png | pbcopy")
}

// swift /Users/larry/coderesp/aphelios_cli/scripts/ocr.swift /Users/larry/Documents/resources/page_32.png | pbcopy
