package com.mjarthur.scanner

import androidx.camera.core.ImageProxy
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer

object CVUtils {
    /**
     * Converts a CameraX ImageProxy (YUV_420_888) to an OpenCV Mat (Grayscale).
     * This is the fastest conversion for edge detection.
     */
    fun imageToGrayMat(image: ImageProxy): Mat {
        val plane = image.planes[0]
        val buffer: ByteBuffer = plane.buffer
        val data = ByteArray(buffer.remaining())
        buffer.get(data)

        val mat = Mat(image.height, image.width, CvType.CV_8UC1)
        mat.put(0, 0, data)
        return mat
    }

    /**
     * Converts a CameraX ImageProxy to an OpenCV Mat (RGBA).
     */
    fun yuvToRgbaMat(image: ImageProxy): Mat {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvMat = Mat(image.height + image.height / 2, image.width, CvType.CV_8UC1)
        yuvMat.put(0, 0, nv21)

        val rgbaMat = Mat()
        Imgproc.cvtColor(yuvMat, rgbaMat, Imgproc.COLOR_YUV2RGBA_NV21, 4)
        return rgbaMat
    }
}