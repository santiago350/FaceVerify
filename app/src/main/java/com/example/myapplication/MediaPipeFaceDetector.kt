package com.example.myapplication

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

data class MediaPipeDetection(
    val confidence: Float,
    val boundingBox: RectF,
    val keypoints: List<PointF>
)

class MediaPipeFaceDetector(context: Context) : Closeable {

    companion object {
        private const val TAG = "FACEVERIFY"
        private const val INPUT_SIZE = 256
        private const val NUM_ANCHORS = 896
        private const val SCORE_THRESHOLD = 0.5f
    }

    private val interpreter: Interpreter

    private val anchors: List<PointF> = generateAnchors()

    init {
        val bytes = context.assets
            .open("MediaPipeFaceDetector.tflite")
            .use { it.readBytes() }

        val modelBuffer = ByteBuffer.allocateDirect(bytes.size)
            .order(ByteOrder.nativeOrder())

        modelBuffer.put(bytes)
        modelBuffer.rewind()

        interpreter = Interpreter(modelBuffer)

        Log.d(TAG, "MediaPipeFaceDetector inicializado")
    }

    fun detect(bitmap: Bitmap): MediaPipeDetection? {

        val input = Array(1) {
            Array(INPUT_SIZE) {
                Array(INPUT_SIZE) {
                    FloatArray(3)
                }
            }
        }

        val scale = min(
            INPUT_SIZE.toFloat() / bitmap.width,
            INPUT_SIZE.toFloat() / bitmap.height
        )

        val scaledWidth = bitmap.width * scale
        val scaledHeight = bitmap.height * scale

        val padX = (INPUT_SIZE - scaledWidth) / 2f
        val padY = (INPUT_SIZE - scaledHeight) / 2f

        val resized = Bitmap.createBitmap(
            INPUT_SIZE,
            INPUT_SIZE,
            Bitmap.Config.ARGB_8888
        )

        val canvas = Canvas(resized)
        canvas.drawColor(Color.BLACK)

        val paint = Paint(Paint.FILTER_BITMAP_FLAG)

        canvas.drawBitmap(
            bitmap,
            null,
            RectF(
                padX,
                padY,
                padX + scaledWidth,
                padY + scaledHeight
            ),
            paint
        )

        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                val pixel = resized.getPixel(x, y)

                input[0][y][x][0] =
                    (Color.red(pixel) / 127.5f) - 1f

                input[0][y][x][1] =
                    (Color.green(pixel) / 127.5f) - 1f

                input[0][y][x][2] =
                    (Color.blue(pixel) / 127.5f) - 1f
            }
        }

        resized.recycle()

        val boxes = Array(1) {
            Array(NUM_ANCHORS) {
                FloatArray(16)
            }
        }

        val scores = Array(1) {
            Array(NUM_ANCHORS) {
                FloatArray(1)
            }
        }

        val outputs = hashMapOf<Int, Any>(
            0 to boxes,
            1 to scores
        )

        interpreter.runForMultipleInputsOutputs(
            arrayOf(input),
            outputs
        )

        var bestIndex = -1
        var bestScore = 0f

        for (i in 0 until NUM_ANCHORS) {
            val rawScore = scores[0][i][0]
                .coerceIn(-80f, 80f)

            val confidence = (
                    1f / (1f + exp(-rawScore))
                    )

            if (confidence > bestScore) {
                bestScore = confidence
                bestIndex = i
            }
        }

        Log.d(
            TAG,
            "MediaPipe confianza máxima: $bestScore"
        )

        if (bestIndex < 0 || bestScore < SCORE_THRESHOLD) {
            Log.d(TAG, "MediaPipe: ningún rostro supera el umbral")
            return null
        }

        val anchor = anchors[bestIndex]
        val raw = boxes[0][bestIndex]

        fun decodeX(index: Int): Float {
            return raw[index] / INPUT_SIZE + anchor.x
        }

        fun decodeY(index: Int): Float {
            return raw[index] / INPUT_SIZE + anchor.y
        }

        val centerX = decodeX(0)
        val centerY = decodeY(1)
        val width = raw[2] / INPUT_SIZE
        val height = raw[3] / INPUT_SIZE

        fun originalX(normalized: Float): Float {
            return ((normalized * INPUT_SIZE - padX) / scale)
                .coerceIn(0f, bitmap.width.toFloat())
        }

        fun originalY(normalized: Float): Float {
            return ((normalized * INPUT_SIZE - padY) / scale)
                .coerceIn(0f, bitmap.height.toFloat())
        }

        val left = originalX(centerX - width / 2f)
        val top = originalY(centerY - height / 2f)
        val right = originalX(centerX + width / 2f)
        val bottom = originalY(centerY + height / 2f)

        val keypoints = mutableListOf<PointF>()

        for (point in 0 until 6) {
            val xIndex = 4 + point * 2
            val yIndex = 5 + point * 2

            keypoints.add(
                PointF(
                    originalX(decodeX(xIndex)),
                    originalY(decodeY(yIndex))
                )
            )
        }

        val result = MediaPipeDetection(
            confidence = bestScore,
            boundingBox = RectF(left, top, right, bottom),
            keypoints = keypoints
        )

        Log.d(TAG, "MediaPipe caja: ${result.boundingBox}")
        Log.d(TAG, "MediaPipe puntos: ${result.keypoints}")

        return result
    }

    private fun generateAnchors(): List<PointF> {

        val result = mutableListOf<PointF>()

        val strides = listOf(16, 32, 32, 32)

        var layer = 0

        while (layer < strides.size) {

            var lastLayer = layer

            while (
                lastLayer < strides.size &&
                strides[lastLayer] == strides[layer]
            ) {
                lastLayer++
            }

            val repeats = (lastLayer - layer) * 2
            val stride = strides[layer]

            val featureHeight = INPUT_SIZE / stride
            val featureWidth = INPUT_SIZE / stride

            for (y in 0 until featureHeight) {
                for (x in 0 until featureWidth) {

                    val centerX = (x + 0.5f) / featureWidth
                    val centerY = (y + 0.5f) / featureHeight

                    repeat(repeats) {
                        result.add(PointF(centerX, centerY))
                    }
                }
            }

            layer = lastLayer
        }

        require(result.size == NUM_ANCHORS) {
            "Cantidad de anclas incorrecta: ${result.size}"
        }

        Log.d(TAG, "Anclas generadas: ${result.size}")

        return result
    }

    override fun close() {
        interpreter.close()
        Log.d(TAG, "MediaPipeFaceDetector cerrado")
    }
}
