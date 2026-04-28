#!/usr/bin/env swift

import Foundation
import AppKit

// macOS icon corner radius ratio: ~22.5% of size
let cornerRadiusRatio: CGFloat = 0.225

// Standard .iconset sizes (width x height)
// @2x means 2x the base size in pixels
let iconSizes: [(name: String, size: CGFloat)] = [
    ("icon_16x16", 16),
    ("icon_16x16@2x", 32),
    ("icon_32x32", 32),
    ("icon_32x32@2x", 64),
    ("icon_128x128", 128),
    ("icon_128x128@2x", 256),
    ("icon_256x256", 256),
    ("icon_256x256@2x", 512),
    ("icon_512x512", 512),
    ("icon_512x512@2x", 1024),
]

func createRoundedRectImage(from image: NSImage, size: NSSize) -> NSImage? {
    let resultImage = NSImage(size: size)
    let cornerRadius = size.width * cornerRadiusRatio
    let rect = NSRect(origin: .zero, size: size)

    resultImage.lockFocus()

    // Draw rounded rect path
    let path = NSBezierPath(roundedRect: rect, xRadius: cornerRadius, yRadius: cornerRadius)
    path.addClip()

    // Draw the source image scaled to fit
    image.draw(in: rect, from: .zero, operation: .copy, fraction: 1.0)

    resultImage.unlockFocus()

    return resultImage
}

func convertToPNGData(_ image: NSImage) -> Data? {
    guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        return nil
    }
    let bitmapRep = NSBitmapImageRep(cgImage: cgImage)
    bitmapRep.size = image.size
    return bitmapRep.representation(using: .png, properties: [:])
}

// --- Main ---

let args = CommandLine.arguments
guard args.count >= 3 else {
    print("Usage: generate_rounded_icon.swift <input.png> <output.icns>")
    exit(1)
}

let inputPath = args[1]
let outputPath = args[2]

guard let originalImage = NSImage(contentsOfFile: inputPath) else {
    print("Error: Cannot load image from \(inputPath)")
    exit(1)
}

// Create temporary .iconset directory
let iconsetPath = (outputPath as NSString).deletingPathExtension + ".iconset"
try? FileManager.default.removeItem(atPath: iconsetPath)
try? FileManager.default.createDirectory(atPath: iconsetPath, withIntermediateDirectories: true)

for iconSize in iconSizes {
    let pixelSize = NSSize(width: iconSize.size, height: iconSize.size)
    guard let roundedImage = createRoundedRectImage(from: originalImage, size: pixelSize),
          let pngData = convertToPNGData(roundedImage) else {
        print("Error: Failed to process size \(iconSize.name)")
        exit(1)
    }

    let filePath = "\(iconsetPath)/\(iconSize.name).png"
    try? pngData.write(to: URL(fileURLWithPath: filePath))
    print("  ✓ \(iconSize.name).png (\(Int(iconSize.size))x\(Int(iconSize.size)))")
}

// Convert .iconset to .icns using iconutil
let process = Process()
process.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
process.arguments = ["-c", "icns", iconsetPath, "-o", outputPath]

let pipe = Pipe()
process.standardError = pipe
process.standardOutput = pipe

try process.run()
process.waitUntilExit()

if process.terminationStatus == 0 {
    print("\n✅ Generated: \(outputPath)")
    // Clean up iconset
    try? FileManager.default.removeItem(atPath: iconsetPath)
} else {
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    let errorMsg = String(data: data, encoding: .utf8) ?? "Unknown error"
    print("Error: iconutil failed: \(errorMsg)")
    exit(1)
}
