
package com.example.myapplication

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.metadata.MetadataExtractor
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

class FaceEmbeddingProcessor(context: Context) : Closeable {

    companion object {
        private const val TAG = "FACEVERIFY"
        private const val INPUT_SIZE = 112
        private const val EMBEDDING_SIZE = 128
        private const val EPSILON = 1e-12
    }

    private val interpreter: Interpreter

    init {
        val modelBytes = context.assets
            .open("mobile_facenet.tflite")
            .use { it.readBytes() }

        val modelBuffer = ByteBuffer.allocateDirect(modelBytes.size)
            .order(ByteOrder.nativeOrder())

        modelBuffer.put(modelBytes)
        modelBuffer.rewind()

        interpreter = Interpreter(modelBuffer)

        Log.d(TAG, "========== MODELO FACIAL ==========")

        Log.d(TAG, "Entradas: ${interpreter.inputTensorCount}")

        for (i in 0 until interpreter.inputTensorCount) {
            val tensor = interpreter.getInputTensor(i)

            Log.d(
                TAG,
                "Entrada $i: shape=${tensor.shape().contentToString()}, " +
                        "tipo=${tensor.dataType()}"
            )
        }

        Log.d(TAG, "Salidas: ${interpreter.outputTensorCount}")

        for (i in 0 until interpreter.outputTensorCount) {
            val tensor = interpreter.getOutputTensor(i)

            Log.d(
                TAG,
                "Salida $i: shape=${tensor.shape().contentToString()}, " +
                        "tipo=${tensor.dataType()}"
            )
        }

        validateModelStructure()

        try {
            val metadataBuffer = ByteBuffer.allocateDirect(modelBytes.size)
                .order(ByteOrder.nativeOrder())

            metadataBuffer.put(modelBytes)
            metadataBuffer.rewind()

            val metadata = MetadataExtractor(metadataBuffer)

            Log.d(
                TAG,
                "¿El modelo contiene metadata?: ${metadata.hasMetadata()}"
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error leyendo metadata: ${e.message}", e)
        }

        Log.d(TAG, "MobileFaceNet inicializado correctamente.")
    }

    private fun validateModelStructure() {
        require(interpreter.inputTensorCount == 2) {
            "El modelo debe tener exactamente dos entradas."
        }

        require(interpreter.outputTensorCount == 1) {
            "El modelo debe tener exactamente una salida."
        }

        val expectedInputShape = intArrayOf(1, 3, INPUT_SIZE, INPUT_SIZE)

        for (i in 0 until interpreter.inputTensorCount) {
            val tensor = interpreter.getInputTensor(i)

            require(tensor.shape().contentEquals(expectedInputShape)) {
                "La entrada $i tiene una dimensión inesperada: " +
                        tensor.shape().contentToString()
            }

            require(tensor.dataType() == DataType.FLOAT32) {
                "La entrada $i debe ser FLOAT32."
            }
        }

        val outputTensor = interpreter.getOutputTensor(0)

        require(
            outputTensor.shape().contentEquals(
                intArrayOf(2, EMBEDDING_SIZE)
            )
        ) {
            "La salida del modelo debe tener dimensiones [2, 128]. " +
                    "Dimensiones encontradas: " +
                    outputTensor.shape().contentToString()
        }

        require(outputTensor.dataType() == DataType.FLOAT32) {
            "La salida del modelo debe ser FLOAT32."
        }
    }

    fun generateEmbedding(bitmap: Bitmap): FloatArray {
        return generateEmbeddings(bitmap, bitmap).first
    }

    fun generateEmbeddings(
        firstBitmap: Bitmap,
        secondBitmap: Bitmap
    ): Pair<FloatArray, FloatArray> {

        require(!firstBitmap.isRecycled) {
            "La primera imagen está liberada."
        }

        require(!secondBitmap.isRecycled) {
            "La segunda imagen está liberada."
        }

        val input1 = bitmapToInputBuffer(firstBitmap)
        val input2 = bitmapToInputBuffer(secondBitmap)

        val output = Array(2) {
            FloatArray(EMBEDDING_SIZE)
        }

        val inputs = arrayOf<Any>(
            input1,
            input2
        )

        val outputs = hashMapOf<Int, Any>(
            0 to output
        )

        interpreter.runForMultipleInputsOutputs(inputs, outputs)

        // Validamos y normalizamos cada embedding.
        val embedding1 = normalizeEmbedding(output[0])
        val embedding2 = normalizeEmbedding(output[1])

        Log.d(
            TAG,
            "EMBEDDING 1 NORMALIZADO: ${embedding1.take(10)}"
        )

        Log.d(
            TAG,
            "EMBEDDING 2 NORMALIZADO: ${embedding2.take(10)}"
        )

        Log.d(
            TAG,
            "Norma embedding 1: ${calculateNorm(embedding1)}"
        )

        Log.d(
            TAG,
            "Norma embedding 2: ${calculateNorm(embedding2)}"
        )

        return Pair(
            embedding1,
            embedding2
        )
    }

    private fun normalizeEmbedding(
        embedding: FloatArray
    ): FloatArray {

        require(embedding.size == EMBEDDING_SIZE) {
            "El embedding debe contener exactamente 128 valores."
        }

        var sumSquares = 0.0

        for (value in embedding) {
            require(value.isFinite()) {
                "El embedding contiene valores no válidos."
            }

            val number = value.toDouble()
            sumSquares += number * number
        }

        val norm = sqrt(sumSquares)

        require(norm.isFinite() && norm > EPSILON) {
            "No se puede normalizar un embedding vacío o inválido."
        }

        return FloatArray(embedding.size) { index ->
            (embedding[index].toDouble() / norm).toFloat()
        }
    }

    private fun calculateNorm(
        embedding: FloatArray
    ): Double {
        var sum = 0.0

        for (value in embedding) {
            sum += value.toDouble() * value.toDouble()
        }

        return sqrt(sum)
    }

    private fun bitmapToInputBuffer(
        bitmap: Bitmap
    ): ByteBuffer {

        require(bitmap.width > 0 && bitmap.height > 0) {
            "La imagen tiene dimensiones inválidas."
        }

        val resized = Bitmap.createScaledBitmap(
            bitmap,
            INPUT_SIZE,
            INPUT_SIZE,
            true
        )

        try {
            val buffer = ByteBuffer.allocateDirect(
                1 * 3 * INPUT_SIZE * INPUT_SIZE * 4
            ).order(ByteOrder.nativeOrder())

            // Conservamos el formato original:
            // RGB, FLOAT32, rango [0, 1].
            // Orden de memoria: NCHW [1, 3, 112, 112].

            for (channel in 0 until 3) {
                for (y in 0 until INPUT_SIZE) {
                    for (x in 0 until INPUT_SIZE) {

                        val pixel = resized.getPixel(x, y)

                        val value = when (channel) {
                            0 -> Color.red(pixel)
                            1 -> Color.green(pixel)
                            else -> Color.blue(pixel)
                        }

                        buffer.putFloat(value / 255.0f)
                    }
                }
            }

            buffer.rewind()

            return buffer

        } finally {
            if (resized !== bitmap && !resized.isRecycled) {
                resized.recycle()
            }
        }
    }

    override fun close() {
        interpreter.close()
        Log.d(TAG, "MobileFaceNet cerrado.")
    }
}
