package com.mjarthur.scanner

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.mjarthur.scanner.databinding.ActivityMainBinding
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService

    // 1. The ImageCapture object for high-res photos
    private var imageCapture: ImageCapture? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV initialized successfully")
        } else {
            Log.e(TAG, "OpenCV initialization failed")
        }

        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        // 2. Wire up the physical capture button
        viewBinding.captureButton.setOnClickListener {
            takePhotoAndWarp()
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        try {
                            val grayMat = CVUtils.imageToGrayMat(imageProxy)

                            val rotatedMat = Mat()
                            when (imageProxy.imageInfo.rotationDegrees) {
                                90 -> Core.rotate(grayMat, rotatedMat, Core.ROTATE_90_CLOCKWISE)
                                180 -> Core.rotate(grayMat, rotatedMat, Core.ROTATE_180)
                                270 -> Core.rotate(grayMat, rotatedMat, Core.ROTATE_90_COUNTERCLOCKWISE)
                                else -> grayMat.copyTo(rotatedMat)
                            }

                            val edges = Mat()
                            Imgproc.Canny(rotatedMat, edges, 75.0, 200.0)
                            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
                            Imgproc.dilate(edges, edges, kernel)
                            Imgproc.erode(edges, edges, kernel)
                            kernel.release()

                            val contours = mutableListOf<MatOfPoint>()
                            val hierarchy = Mat()
                            Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

                            contours.sortByDescending { Imgproc.contourArea(it) }
                            var largestQuad: List<Point>? = null

                            val minArea = rotatedMat.width() * rotatedMat.height() * 0.05

                            for (contour in contours.take(5)) {
                                val area = Imgproc.contourArea(contour)
                                if (area > minArea) {
                                    val m2f = MatOfPoint2f(*contour.toArray())
                                    val approx = MatOfPoint2f()
                                    val perimeter = Imgproc.arcLength(m2f, true)

                                    Imgproc.approxPolyDP(m2f, approx, 0.02 * perimeter, true)

                                    if (approx.total() == 4L && Imgproc.isContourConvex(MatOfPoint(*approx.toArray()))) {
                                        val points = approx.toList()

                                        val tl = points.minByOrNull { p -> p.x + p.y }!!
                                        val br = points.maxByOrNull { p -> p.x + p.y }!!
                                        val tr = points.minByOrNull { p -> p.y - p.x }!!
                                        val bl = points.maxByOrNull { p -> p.y - p.x }!!

                                        largestQuad = listOf(tl, tr, br, bl)
                                        break
                                    }
                                }
                            }

                            val overlayWidth = viewBinding.quadOverlay.width.toDouble()
                            val overlayHeight = viewBinding.quadOverlay.height.toDouble()
                            val matWidth = rotatedMat.width().toDouble()
                            val matHeight = rotatedMat.height().toDouble()

                            if (largestQuad != null && overlayWidth > 0 && matWidth > 0) {
                                val scale = Math.max(overlayWidth / matWidth, overlayHeight / matHeight)
                                val offsetX = (overlayWidth - matWidth * scale) / 2.0
                                val offsetY = (overlayHeight - matHeight * scale) / 2.0

                                val scaledPoints = largestQuad.map { p ->
                                    Point(p.x * scale + offsetX, p.y * scale + offsetY)
                                }

                                runOnUiThread {
                                    viewBinding.quadOverlay.setPoints(scaledPoints)
                                }
                            } else {
                                runOnUiThread {
                                    viewBinding.quadOverlay.setPoints(null)
                                }
                            }

                            grayMat.release()
                            rotatedMat.release()
                            edges.release()
                            hierarchy.release()

                        } catch (e: Exception) {
                            Log.e("CV_DEBUG", "CRASH in OpenCV Math: ${e.message}")
                        } finally {
                            imageProxy.close()
                        }
                    }
                }

            // 3. Initialize High-Res Capture
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .build()

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                // Bind Preview, Capture, AND Analyzer
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture, imageAnalyzer
                )
            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    // 4. Capture & Warp Logic
    private fun takePhotoAndWarp() {
        val imageCapture = imageCapture ?: return

        // Grab coordinates immediately to avoid Tap Shake
        val lockedPoints = viewBinding.quadOverlay.points
        if (lockedPoints == null || lockedPoints.size != 4) {
            Toast.makeText(this, "No document detected! Hold still.", Toast.LENGTH_SHORT).show()
            return
        }

        val overlayWidth = viewBinding.quadOverlay.width.toDouble()
        val overlayHeight = viewBinding.quadOverlay.height.toDouble()

        Toast.makeText(this, "Capturing...", Toast.LENGTH_SHORT).show()

        imageCapture.takePicture(
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(imageProxy: ImageProxy) {
                    super.onCaptureSuccess(imageProxy)

                    try {
                        val buffer = imageProxy.planes[0].buffer
                        val bytes = ByteArray(buffer.remaining())
                        buffer.get(bytes)

                        val originalMat = Imgcodecs.imdecode(MatOfByte(*bytes), Imgcodecs.IMREAD_COLOR)

                        val rotatedMat = Mat()
                        when (imageProxy.imageInfo.rotationDegrees) {
                            90 -> Core.rotate(originalMat, rotatedMat, Core.ROTATE_90_CLOCKWISE)
                            180 -> Core.rotate(originalMat, rotatedMat, Core.ROTATE_180)
                            270 -> Core.rotate(originalMat, rotatedMat, Core.ROTATE_90_COUNTERCLOCKWISE)
                            else -> originalMat.copyTo(rotatedMat)
                        }

                        val matWidth = rotatedMat.width().toDouble()
                        val matHeight = rotatedMat.height().toDouble()

                        val scale = Math.max(overlayWidth / matWidth, overlayHeight / matHeight)
                        val offsetX = (overlayWidth - matWidth * scale) / 2.0
                        val offsetY = (overlayHeight - matHeight * scale) / 2.0

                        val highResPoints = lockedPoints.map { p ->
                            Point((p.x - offsetX) / scale, (p.y - offsetY) / scale)
                        }

                        val scannedDocumentMat = warpDocument(rotatedMat, highResPoints)

                        saveMatToGallery(scannedDocumentMat)

                        scannedDocumentMat.release()
                        originalMat.release()
                        rotatedMat.release()

                    } catch (e: Exception) {
                        Log.e("CV_CAPTURE", "Failed to process image: ${e.message}")
                    } finally {
                        imageProxy.close()
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e("CV_CAPTURE", "Photo capture failed: ${exception.message}", exception)
                }
            }
        )
    }

    // 5. Homography/Perspective Warp Math
    private fun warpDocument(sourceMat: Mat, points: List<Point>): Mat {
        val srcPoints = MatOfPoint2f(*points.toTypedArray())

        val widthA = Math.hypot(points[2].x - points[3].x, points[2].y - points[3].y)
        val widthB = Math.hypot(points[1].x - points[0].x, points[1].y - points[0].y)
        val maxWidth = Math.max(widthA, widthB).toFloat()

        val heightA = Math.hypot(points[1].x - points[2].x, points[1].y - points[2].y)
        val heightB = Math.hypot(points[0].x - points[3].x, points[0].y - points[3].y)
        val maxHeight = Math.max(heightA, heightB).toFloat()

        val dstPoints = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(maxWidth.toDouble() - 1, 0.0),
            Point(maxWidth.toDouble() - 1, maxHeight.toDouble() - 1),
            Point(0.0, maxHeight.toDouble() - 1)
        )

        val perspectiveTransform = Imgproc.getPerspectiveTransform(srcPoints, dstPoints)
        val warpedMat = Mat()
        Imgproc.warpPerspective(
            sourceMat,
            warpedMat,
            perspectiveTransform,
            Size(maxWidth.toDouble(), maxHeight.toDouble())
        )

        srcPoints.release()
        dstPoints.release()
        perspectiveTransform.release()

        return warpedMat
    }

    // 6. Save Mat to Camera Roll
    private fun saveMatToGallery(mat: Mat) {
        try {
            val rgbMat = Mat()
            Imgproc.cvtColor(mat, rgbMat, Imgproc.COLOR_BGR2RGB)

            val bitmap = android.graphics.Bitmap.createBitmap(rgbMat.cols(), rgbMat.rows(), android.graphics.Bitmap.Config.ARGB_8888)
            org.opencv.android.Utils.matToBitmap(rgbMat, bitmap)

            val contentValues = android.content.ContentValues().apply {
                put(android.provider.MediaStore.MediaColumns.DISPLAY_NAME, "Scan_${System.currentTimeMillis()}.jpg")
                put(android.provider.MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
                put(android.provider.MediaStore.MediaColumns.RELATIVE_PATH, android.os.Environment.DIRECTORY_PICTURES + "/Scans")
            }

            val uri = contentResolver.insert(android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
            if (uri != null) {
                contentResolver.openOutputStream(uri).use { out ->
                    bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 100, out!!)
                }
                runOnUiThread {
                    Toast.makeText(this, "Saved to Camera Roll!", Toast.LENGTH_LONG).show()
                }
            }

            rgbMat.release()
        } catch (e: Exception) {
            Log.e("CV_CAPTURE", "Error saving to gallery: ${e.message}")
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "Permissions not granted.", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "ScannerApp"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}