package com.mjarthur.scanner

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View
import org.opencv.core.Point

class QuadOverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    // 1. The private backing variable (only this file can change it directly)
    private var _points: List<Point>? = null

    // 2. A public read-only property so MainActivity can safely look at the points
    val points: List<Point>?
        get() = _points

    private val paint = Paint().apply {
        color = Color.GREEN
        strokeWidth = 15f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    // 3. Our manual setter function
    fun setPoints(newPoints: List<Point>?) {
        _points = newPoints
        postInvalidate() // Forces the UI to redraw safely
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val currentPoints = _points
        if (currentPoints == null || currentPoints.size != 4) {
            return
        }

        val path = Path()
        path.moveTo(currentPoints[0].x.toFloat(), currentPoints[0].y.toFloat())
        path.lineTo(currentPoints[1].x.toFloat(), currentPoints[1].y.toFloat())
        path.lineTo(currentPoints[2].x.toFloat(), currentPoints[2].y.toFloat())
        path.lineTo(currentPoints[3].x.toFloat(), currentPoints[3].y.toFloat())
        path.close()

        canvas.drawPath(path, paint)
    }
}