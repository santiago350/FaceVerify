package com.example.myapplication

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder

object ModelDiagnostics {

    fun inspect(context: Context, assetName: String) {
        var interpreter: Interpreter? = null

        try {
            val modelBytes = context.assets
                .open(assetName)
                .use { it.readBytes() }

            val modelBuffer = ByteBuffer
                .allocateDirect(modelBytes.size)
                .order(ByteOrder.nativeOrder())

            modelBuffer.put(modelBytes)
            modelBuffer.rewind()

            interpreter = Interpreter(modelBuffer)

            Log.d("FACEVERIFY", "========== $assetName ==========")
            Log.d("FACEVERIFY", "Tamaño: ${modelBytes.size} bytes")
            Log.d(
                "FACEVERIFY",
                "Entradas: ${interpreter.inputTensorCount}"
            )
            Log.d(
                "FACEVERIFY",
                "Salidas: ${interpreter.outputTensorCount}"
            )

            for (i in 0 until interpreter.inputTensorCount) {
                val tensor = interpreter.getInputTensor(i)

                Log.d(
                    "FACEVERIFY",
                    "INPUT $i: shape=${tensor.shape().contentToString()}, " +
                            "tipo=${tensor.dataType()}, " +
                            "bytes=${tensor.numBytes()}"
                )
            }

            for (i in 0 until interpreter.outputTensorCount) {
                val tensor = interpreter.getOutputTensor(i)

                Log.d(
                    "FACEVERIFY",
                    "OUTPUT $i: shape=${tensor.shape().contentToString()}, " +
                            "tipo=${tensor.dataType()}, " +
                            "bytes=${tensor.numBytes()}"
                )
            }

        } catch (e: Exception) {
            Log.e(
                "FACEVERIFY",
                "Error inspeccionando $assetName",
                e
            )
        } finally {
            interpreter?.close()
        }
    }
}
