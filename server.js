// ========================================
// TensorFlow backend auto-detect
// ========================================

let tf;
let backendName = "unknown";

try {
  // Try fast native backend first
  tf = require("@tensorflow/tfjs-node");
  backendName = "tfjs-node (FAST)";
} catch (err) {
  // Fallback to pure JS
  tf = require("@tensorflow/tfjs");
  backendName = "tfjs (JS fallback)";
}

console.log("🧠 TensorFlow backend:", backendName);

// ========================================
// Normal imports
// ========================================

const express = require("express");
const multer = require("multer");
const cors = require("cors");
const faceapi = require("face-api.js");
const canvas = require("canvas");
const fetch = require("node-fetch");
const path = require("path");

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app = express();
app.use(cors());
app.use(express.json());

// =============================
// CONFIG (tune here)
// =============================
const FACE_MATCH_THRESHOLD = 0.48;
const MIN_FACE_WIDTH = 120;
const MIN_IMAGE_SIZE = 300;
const INPUT_SIZE = 512;

// =============================
// Multer (memory only)
// =============================
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 5 * 1024 * 1024 },
});

// =============================
// Load Models (once)
// =============================
let modelsLoaded = false;

async function loadModels() {
  if (modelsLoaded) return;

  const modelPath = path.join(__dirname, "models");

  console.log("🔄 Loading face models...");

  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);

  modelsLoaded = true;
  console.log("✅ Face models loaded");
}

// =============================
// Helpers
// =============================

// URL → Image
async function loadImageFromUrl(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("Failed to fetch reference image");
  const buffer = await res.buffer();
  return await canvas.loadImage(buffer);
}

// Buffer → Image
async function loadImageFromBuffer(buffer) {
  return await canvas.loadImage(buffer);
}

// =============================
// HIGH ACCURACY face extraction
// =============================
async function extractFaceData(image) {
  await loadModels();

  // ❌ low resolution
  if (image.width < MIN_IMAGE_SIZE || image.height < MIN_IMAGE_SIZE) {
    return {
      success: false,
      msg: "Image too low quality. Use better camera.",
    };
  }

  const detections = await faceapi
    .detectAllFaces(
      image,
      new faceapi.SsdMobilenetv1Options({
        minConfidence: 0.5,
        maxResults: 5,
        inputSize: INPUT_SIZE,
      })
    )
    .withFaceLandmarks()
    .withFaceDescriptors();

  // ❌ no face
  if (!detections || detections.length === 0) {
    return { success: false, msg: "No face detected" };
  }

  // ❌ multiple faces
  if (detections.length > 1) {
    return { success: false, msg: "Multiple faces detected" };
  }

  const faceBox = detections[0].detection.box;

  // ❌ face too small
  if (faceBox.width < MIN_FACE_WIDTH) {
    return {
      success: false,
      msg: "Face too small. Move closer to camera.",
    };
  }

  return {
    success: true,
    descriptor: detections[0].descriptor,
  };
}

// =============================
// Compare faces + percentage
// =============================
function compareFaces(d1, d2) {
  const distance = faceapi.euclideanDistance(d1, d2);

  // convert to percentage
  let matchPercentage = (1 - distance) * 100;
  matchPercentage = Math.max(0, Math.min(100, matchPercentage));

  return {
    verified: distance < FACE_MATCH_THRESHOLD,
    distance: Number(distance.toFixed(4)),
    matchPercentage: Number(matchPercentage.toFixed(2)),
    threshold: FACE_MATCH_THRESHOLD,
  };
}

// =============================
// Health check
// =============================
app.get("/", (req, res) => {
  res.json({
    service: "Tadipaar Face Verification Service",
    status: "running",
    accuracyMode: "HIGH",
  });
});

// =============================
// MAIN VERIFY ENDPOINT
// =============================
app.post("/verify", upload.single("selfie"), async (req, res) => {
  try {
    const { referenceImageUrl } = req.body;
    const selfieFile = req.file;

    if (!referenceImageUrl || !selfieFile) {
      return res.status(400).json({
        msg: "referenceImageUrl and selfie file required",
      });
    }

    // 🔹 Reference image
    const referenceImg = await loadImageFromUrl(referenceImageUrl);
    const refData = await extractFaceData(referenceImg);

    if (!refData.success) {
      return res.status(400).json({
        msg: "Reference image error: " + refData.msg,
      });
    }

    // 🔹 Selfie image
    const selfieImg = await loadImageFromBuffer(selfieFile.buffer);
    const selfieData = await extractFaceData(selfieImg);

    if (!selfieData.success) {
      return res.status(400).json({
        msg: "Selfie error: " + selfieData.msg,
      });
    }

    // 🔹 Compare
    const result = compareFaces(
      refData.descriptor,
      selfieData.descriptor
    );

    return res.json(result);
  } catch (err) {
    console.error("❌ Verification error:", err);
    return res.status(500).json({
      msg: "Face verification failed",
    });
  }
});

// =============================
// Start server
// =============================
const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log(`🚀 Face service running on port ${PORT}`);
});